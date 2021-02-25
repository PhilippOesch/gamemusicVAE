import warnings
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from utility.mode_collapse import*
import os

num_notes = 88
min_tresh= 0.01
similarity_threshhold = 0.01

warnings.filterwarnings("ignore", category=RuntimeWarning)


def write_evaluation(file_name, modelname, title, metric_labels, values):
    write_dir = '../evaluation_results/' + modelname + '/'
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    with open(write_dir + file_name, 'w') as f:
        f.write(title + "\n\n")

        for metric, value in zip(metric_labels, values):
            f.write(metric + ": " + str(value) + "\n")


def write_val_to_train_comparrison(file_name, title, dictionary, lowestTD):
    write_dir = '../evaluation_results/'
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    with open(write_dir + file_name, 'w') as f:
        f.write(title + "\n\n")

        avarage = 0
        for key in dictionary:
            f.write(str(key) + ": " + str(dictionary[key]) + "\n")
            avarage += dictionary[key]

        avarage /= len(dictionary)

        f.write("Avarage Tonal Distance: " + str(avarage) + "\n")
        f.write("Lowest Tonal Distance between 2 Songs: " + str(lowestTD) + "\n")

def get_qn(samples, thresh=0.25):
    note_counter = 0
    qualified_note_counter = 0

    adjusted_samples = np.reshape(
        samples, (samples.shape[0] * samples.shape[1], samples.shape[2]))

    length_vector = np.zeros((num_notes), dtype=np.uint8)
    for i in range(adjusted_samples.shape[0]):
        for z in range(adjusted_samples.shape[1]):
            if i == adjusted_samples.shape[0]-1:
                if adjusted_samples[i, z] > thresh:
                    length_vector[z] += 1
                if length_vector[z] >= 1:
                    note_counter += 1
                if length_vector[z] >= 3:
                    qualified_note_counter += 1

            if adjusted_samples[i, z] > thresh:
                length_vector[z] += 1
            else:
                if length_vector[z] >= 1:
                    note_counter += 1
                if length_vector[z] >= 3:
                    qualified_note_counter += 1
                length_vector[z] = 0

    ratio = qualified_note_counter / note_counter
    return ratio, qualified_note_counter, note_counter

def get_note_count(samples, thresh= 0.25):
    note_counter= 0

    for sample in samples:
        for y in range(sample.shape[0]):
            for x in range(sample.shape[1]):
                if sample[y, x]> thresh:
                    note_counter+= 1;

    return note_counter



def hcdf(samples, ignore_treshhold=True, thresh=0.5):
    #     # 12 Tone Equal Temperament
    eq_t_samples = samples_to_pitchclasses(samples, ignore_treshhold, thresh)

     # 6d Vectors for tonal centroid calculation
    t_c_samples = get_t_c_samples(samples= eq_t_samples)

    # convolve signal with gaussian
    gaussian = signal.gaussian(100, std=8)
    filter = np.zeros((t_c_samples.shape[0], t_c_samples.shape[1]))


    # Convolve each row
    for i in range(6):
        filter[:, i]= signal.convolve(t_c_samples[:, i], gaussian, mode= 'same')

    hcdf_results = []

    for i in range(1, filter.shape[0]-1):
        epsilon = np.sqrt(np.sum((filter[i-1] - filter[i+1])**2))
        hcdf_results.append(epsilon)

    return hcdf_results

def samples_to_pitchclasses(samples, ignore_treshhold=True, thresh=0.5):
    eq_t_samples = np.zeros(
        (samples.shape[0], samples.shape[1], 12), dtype=np.float32)

    for z in range(samples.shape[0]):
        for y in range(samples.shape[1]):
            note = np.zeros((12), dtype=np.float32)
            for x in range(samples.shape[2]):
                i = x % 12 
                # print(sample[y, x].dtype)
                if ignore_treshhold and samples[z, y, x] > note[i]:
                    note[i] = samples[z, y, x]
                elif not ignore_treshhold and samples[z, y, x] > thresh:
                    note[i] = 1
            eq_t_samples[z, y] = note

    return eq_t_samples

def get_t_c_samples(samples):
    t_c_samples = np.zeros(
        (samples.shape[0], samples.shape[1], 6), dtype=np.float32)

    for z in range(samples.shape[0]):
        for y in range(samples.shape[1]):
            y_d6 = [get_transformation_matrix(x) * samples[z, y, x] for x in range(samples.shape[2])]
            sum_y = np.sum(y_d6, axis=0)
            # print(sum_y)
            l_norm = np.linalg.norm(samples[z, y])
            zeta_y = sum_y * (1 / l_norm)
            zeta_y = np.nan_to_num(zeta_y)
            t_c_samples[z, y] = zeta_y

    t_c_samples = np.reshape(t_c_samples, ((
        t_c_samples.shape[0] * t_c_samples.shape[1]), t_c_samples.shape[2]))

    return t_c_samples

def empty_bars(samples, thresh=0.1):
    empty_bar_count = 0
    for z in range(samples.shape[0]):
        is_empty = True
        for y in range(samples.shape[1]):
            for x in range(samples.shape[2]):
                if samples[z, y, x] >= thresh:
                    is_empty = False
                    continue
        if is_empty:
            empty_bar_count += 1

    return empty_bar_count, samples.shape[0]

# Number of used pitch classes (12 tonen equal temperament)


def get_upc(samples, thresh=0.25):
    eq_t_samples = samples_to_pitchclasses(
        samples=samples, ignore_treshhold=False, thresh=thresh)

    upcs = []
    for z in range(eq_t_samples.shape[0]):
        pitch_classes = np.zeros((12), dtype=np.uint8)
        for y in range(eq_t_samples.shape[1]):
            for x in range(eq_t_samples.shape[2]):
                if pitch_classes[x] < eq_t_samples[z, y, x]:
                    pitch_classes[x] = int(eq_t_samples[z, y, x])

        upcs.append(np.sum(pitch_classes))
    average_upcs = np.average(upcs)

    return upcs, average_upcs


def polyphonicity(sample, thresh=0.25):
    adjusted_samples = np.zeros(
        (sample.shape[0], sample.shape[1], sample.shape[2]), dtype=np.uint8)
    for z in range(sample.shape[0]):
        for y in range(sample.shape[1]):
            for x in range(sample.shape[2]):
                if sample[z, y, x] > thresh:
                    adjusted_samples[z, y, x] = 1

    count_frames_note = 0
    count_frames_multiple_notes = 0

    for z in range(adjusted_samples.shape[0]):
        for y in range(adjusted_samples.shape[1]):
            if np.sum(adjusted_samples[z, y]) > 0:
                count_frames_note += 1
            if np.sum(adjusted_samples[z, y]) > 1:
                count_frames_multiple_notes += 1

    return count_frames_multiple_notes, count_frames_note


def get_transformation_matrix(l):
    r1 = 1
    r2 = 1
    r3 = 0.5

    t_matrix = np.array(
        (r1 * np.sin(l * (7*np.pi / 6)),
         r1 * np.cos(l * (7*np.pi / 6)),
         r2 * np.sin(l * (3*np.pi / 2)),
         r2 * np.cos(l * (3*np.pi / 2)),
         r3 * np.sin(l * (2*np.pi / 3)),
         r3 * np.cos(l * (2*np.pi / 3))
         ))

    return t_matrix

def tonal_distance(samples1, samples2, ignore_treshhold, thresh= 0.5):
        #     # 12 Tone Equal Temperament
    eq_t_samples1 = samples_to_pitchclasses(samples1, ignore_treshhold, thresh)
    eq_t_samples2 = samples_to_pitchclasses(samples2, ignore_treshhold, thresh)

#     # 6d Vectors for tonal centroid calculation
    t_c_samples1 = get_t_c_samples(samples= eq_t_samples1)
    t_c_samples2 = get_t_c_samples(samples= eq_t_samples2)
    # print(t_c_samples.shape)

    e_d_results = []
    for i in range(t_c_samples1.shape[0]):
        epsilon = np.sqrt(np.sum((t_c_samples1[i] - t_c_samples2[i])**2))
        e_d_results.append(epsilon)

    return np.sum(e_d_results)/ len(e_d_results)

def drum_pattern(samples, thresh= 0.5):
    notes_counter= 0;
    dp_notes_counter= 0;
    for z in range(samples.shape[0]):
        for y in range(samples.shape[1]):
            for x in range(samples.shape[2]):
                if y% 6== 0 and samples[z, y, x]> thresh:
                    dp_notes_counter+= 1
                    notes_counter+= 1
                elif samples[z, y, x]> thresh:
                    notes_counter+= 1

    return dp_notes_counter, notes_counter 

def evaluate_dp(data_set, thresh):
    print("Evaluating rate of drum pattern rhythms in tracks")
    dp_sum = 0
    notes_sum= 0
    for song in data_set:
        dp_notes, notes = drum_pattern(song, thresh)
        dp_sum+= dp_notes
        notes_sum+= notes

    return dp_sum / notes_sum


def evaluate_eb(data_set, thresh):
    print("Evaluating Rate of Empty Bars")
    eb_sum = 0
    for song in data_set:
        eb_value, _ = empty_bars(song, thresh)
        eb_sum += eb_value

    return eb_sum / (data_set.shape[0] * data_set.shape[1])


def evaluate_polyphonicity(data_set, thresh):
    print("Evaluating Polyphonicity of tracks")
    poly_sum = 0
    notes_step_sum = 0
    for song in data_set:
        poly, notes = polyphonicity(song, thresh)
        poly_sum+= poly
        notes_step_sum+= notes

    return poly_sum / notes_step_sum


def evaluate_tonal_distance(data_set1, data_set2=[], thresh=0.25):
    print("Evaluating Tonal Distance between tracks")
    missing_sets = data_set1.copy()
    print(missing_sets.shape)
    total_sum = 0
    comparison_counter = 0
    lowestTD = 1
    try:
        for value in data_set1:
            if not missing_sets.shape[0] == 0:
                if len(data_set2) == 0:
                    for compare_value in missing_sets[1:]:
                        print("Comparison Counter: ", comparison_counter)
                        comparison_counter += 1
                        comparisson_td = tonal_distance(
                            value, compare_value, ignore_treshhold=False, thresh=thresh)
                        if comparisson_td < similarity_threshhold:
                            raise ModeCollapse
                        total_sum += comparisson_td
                        if comparisson_td < lowestTD:
                            lowestTD = comparisson_td
                    missing_sets = np.delete(missing_sets, value, axis=0)
                else:
                    for compare_value in data_set2:
                        print("Comparison Counter: ", comparison_counter)
                        comparison_counter += 1
                        comparisson_td = tonal_distance(
                            value, compare_value, ignore_treshhold=False, thresh=thresh)
                        print(comparisson_td)
                        if comparisson_td < similarity_threshhold:
                            raise ModeCollapse
                        total_sum += comparisson_td
                        if comparisson_td < lowestTD:
                            lowestTD = comparisson_td
        print(comparison_counter)
    except ModeCollapse:
        print("Mode Collapse Exception was thrown. The samples are too similar")

    return total_sum / comparison_counter, lowestTD


def evaluate_upc_average(data_set, thresh):
    print("Evaluate Average Number of pitch classes")
    total_upc = 0
    for song in data_set:
        _, average_upcs = get_upc(song, thresh)
        total_upc += average_upcs

    return total_upc / data_set.shape[0]


def evaluate_notes_per_song(data_set, thresh):
    print("Evaluate Notes per Song")
    total_note_counter = 0

    for song in data_set:
        total_note_counter += get_note_count(song, thresh)

    return total_note_counter / data_set.shape[0]