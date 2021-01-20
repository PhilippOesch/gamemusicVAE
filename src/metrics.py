import warnings
import numpy as np

num_notes = 88

warnings.filterwarnings("ignore",category =RuntimeWarning)

# def tonal_distance(samples, ignore_treshhold= True, thresh=0.5):

#     # tranform to chroma vector of size 12
#     # 12 Tone Equal Temperament
#     eq_t_samples = samples_to_pitchclasses(samples)

#     # 6d Vectors for tonal centroid calculation
#     t_c_samples = np.zeros(
#         (samples.shape[0], samples.shape[1], 6), dtype=np.float32)

#     # print(eq_t_samples)

#     for z in range(eq_t_samples.shape[0]):
#         for y in range(eq_t_samples.shape[1]):
#             y_d6= [get_transformation_matrix(x)* eq_t_samples[z ,y, x] for x in range(eq_t_samples.shape[2])]
#             sum_y= np.sum(y_d6, axis= 0);
#             # print(sum_y)
#             l_norm= np.linalg.norm(eq_t_samples[z, y])
#             zeta_y= sum_y * (1/ l_norm)
#             zeta_y= np.nan_to_num(zeta_y)
#             t_c_samples[z, y] = zeta_y

# Harmonic change detection function

def get_qn(samples, thresh= 0.05):
    note_counter= 0;
    qualified_note_counter= 0;

    adjusted_samples= np.reshape(samples, (samples.shape[0]* samples.shape[1], samples.shape[2]))

    length_vector= np.zeros((num_notes), dtype= np.uint8)
    for i in range(adjusted_samples.shape[0]):
        for z in range(adjusted_samples.shape[1]):
            if i== adjusted_samples.shape[0]-1:
                if adjusted_samples[i, z]> thresh:
                    length_vector[z]+= 1
                if length_vector[z]>= 1:
                    note_counter+=1
                if length_vector[z]>=3:
                    qualified_note_counter+= 1

            if adjusted_samples[i, z]> thresh:
                length_vector[z]+= 1
            else:
                if length_vector[z]>=1:
                    note_counter+= 1
                if length_vector[z]>=3:
                    qualified_note_counter+= 1
                length_vector[z]= 0

    ratio= qualified_note_counter/ note_counter
    return ratio, qualified_note_counter, note_counter


def hcdf(samples, ignore_treshhold= True, thresh=0.5):
#     # 12 Tone Equal Temperament
    eq_t_samples = samples_to_pitchclasses(samples)

#     # 6d Vectors for tonal centroid calculation
    t_c_samples = np.zeros(
        (samples.shape[0], samples.shape[1], 6), dtype=np.float32)

#     # print(eq_t_samples)

    for z in range(eq_t_samples.shape[0]):
        for y in range(eq_t_samples.shape[1]):
            y_d6= [get_transformation_matrix(x)* eq_t_samples[z ,y, x] for x in range(eq_t_samples.shape[2])]
            sum_y= np.sum(y_d6, axis= 0);
            # print(sum_y)
            l_norm= np.linalg.norm(eq_t_samples[z, y])
            zeta_y= sum_y * (1/ l_norm)
            zeta_y= np.nan_to_num(zeta_y)
            t_c_samples[z, y] = zeta_y
    
    t_c_samples= np.reshape(t_c_samples, ((t_c_samples.shape[0]* t_c_samples.shape[1]), t_c_samples.shape[2]))
    print(t_c_samples.shape)

    hcdf_results= []
    for i in range(1, t_c_samples.shape[0]-1):
        epsilon= np.sqrt(np.sum((t_c_samples[i-1]- t_c_samples[i+1])**2))
        hcdf_results.append(epsilon)

    return hcdf_results


def samples_to_pitchclasses(samples, ignore_treshhold=True, thresh= 0.5):
    eq_t_samples = np.zeros(
        (samples.shape[0], samples.shape[1], 12), dtype=np.float32)

    for z in range(samples.shape[0]):
        for y in range(samples.shape[1]):
            note = np.zeros((12), dtype=np.float32)
            for x in range(samples.shape[2]):
                i = (x-2) % 12   # -2 Piano starts at note-A instead of note-C
                # print(sample[y, x].dtype)
                if ignore_treshhold and samples[z, y, x] > note[i]:
                    note[i] = samples[z, y, x]
                elif not ignore_treshhold and samples[z, y, x]> thresh:
                    note[i]= 1
            eq_t_samples[z, y] = note
    
    return eq_t_samples

def empty_bars(samples, thresh= 0.05):
    empty_bar_count= 0
    for z in range(samples.shape[0]):
        is_empty= True
        for y in range(samples.shape[1]):
            for x in range(samples.shape[2]):
                if samples[z, y, x]>= thresh:
                    is_empty= False;
        if is_empty:
            empty_bar_count+= 1;
    
    return empty_bar_count, samples.shape[0]

# Number of used pitch classes (12 tonen equal temperament)
def get_upc(samples, thresh= 0.25):
    eq_t_samples= samples_to_pitchclasses(samples= samples, ignore_treshhold=False, thresh=thresh)

    upcs= []
    for z in range(eq_t_samples.shape[0]):
        pitch_classes= np.zeros((12), dtype= np.uint8) 
        for y in range(eq_t_samples.shape[1]):
            for x in range(eq_t_samples.shape[2]):
                if pitch_classes[x]< eq_t_samples[z, y, x]:
                    pitch_classes[x]= int(eq_t_samples[z, y, x])

        upcs.append(np.sum(pitch_classes))
    average_upcs= np.average(upcs)
    
    return upcs, average_upcs

def polyphonicity(samples, relative_to_notes= True, thresh= 0.25):
    adjusted_samples= np.zeros((samples.shape[0], samples.shape[1], samples.shape[2]), dtype= np.uint8)
    for z in range(samples.shape[0]):
        for y in range(samples.shape[1]):
            for x in range(samples.shape[2]):
                if samples[z, y, x]> thresh:
                    adjusted_samples[z, y, x]= 1
    
    count_frames_note= 0
    count_frames_multiple_notes= 0


    for z in range(adjusted_samples.shape[0]):
        for y in range(adjusted_samples.shape[1]):
            if np.sum(adjusted_samples[z, y])>0:
                count_frames_note+= 1
            if np.sum(adjusted_samples[z, y])>1:
                count_frames_multiple_notes+= 1

    if relative_to_notes:   
        ratio= count_frames_multiple_notes/ count_frames_note
    else:
        ratio= count_frames_multiple_notes/ (samples.shape[0]* samples.shape[1])
    return ratio

def get_transformation_matrix(l):
    r1 = 1
    r2 = 1
    r3 = 0.5

    t_matrix= np.array(
        (r1 * np.sin(l * (7*np.pi / 6)),
         r1 * np.cos(l * (7*np.pi / 6)),
         r2 * np.sin(l * (3*np.pi / 2)),
         r2 * np.cos(l * (3*np.pi / 2)),
         r3 * np.sin(l * (2*np.pi / 3)),
         r3 * np.cos(l * (2*np.pi / 3))
         ))

    return t_matrix
