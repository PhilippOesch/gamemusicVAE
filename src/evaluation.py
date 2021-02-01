import os
import utility.util as util
import numpy as np
import matplotlib.pyplot as plt

import params
import network.musicVAE as musicVAE
import utility.metrics as metrics

metric_lables = [
    "Empty Bars",
    "Drum Pattern",
    "Polyphonicity",
    "Average amount of different pitch classes",
    "Tonal Distance between tracks"
]


def evaluate_dp(data_set, thresh):
    print("Evaluating rate of drum pattern rhythms in tracks")
    dp_sum = 0
    for song in data_set:
        dp_sum += metrics.drum_pattern(song, thresh)

    return dp_sum / data_set.shape[0]


def evaluate_eb(data_set):
    print("Evaluating Rate of Empty Bars")
    eb_sum = 0
    for song in data_set:
        eb_value, _ = metrics.empty_bars(song)
        eb_sum += eb_value

    return eb_sum / (data_set.shape[0] * data_set.shape[1])


def evaluate_polyphonicity(data_set, thresh):
    print("Evaluating Polyphonicity of tracks")
    poly_sum = 0
    for song in data_set:
        poly_sum += metrics.polyphonicity(song, True, thresh)

    return poly_sum / data_set.shape[0]


def evaluate_tonal_distance(data_set, thresh):
    print("Evaluating Tonal Distance between tracks")
    missing_sets = data_set.copy()
    print(missing_sets.shape)
    total_sum = 0
    comparison_counter = 0
    for value in data_set:
        if not missing_sets.shape[0] == 0:
            for compare_value in missing_sets[1:]:
                print("Comparison Counter: ", comparison_counter)
                comparison_counter += 1
                _, comparisson_td = metrics.tonal_distance(
                    value, compare_value)
                total_sum += comparisson_td
            missing_sets = np.delete(missing_sets, value, axis=0)
    print(comparison_counter)

    return total_sum / comparison_counter


def evaluate_upc_average(data_set, thresh):
    print("Evaluate Average Number of pitch classes")
    total_upc = 0
    for song in data_set:
        _, average_upcs = metrics.get_upc(song, thresh)
        total_upc += average_upcs

    return total_upc / data_set.shape[0]

def evaluate_notes_per_song(data_set, thresh):
    print("Evaluate Notes per Song")
    total_note_counter= 0
    
    for song in data_set:
        total_note_counter+= metrics.get_note_count(song, thresh)
    
    return total_note_counter/ data_set.shape[0]

def write_evaluation(file_name, title, metric_labels, values):
    write_dir = '../evaluation_results/overworld_theme/'
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    with open(write_dir + file_name, 'w') as f:
        f.write(title + "\n\n")

        for metric, value in zip(metric_labels, values):
            f.write(metric + ": " + str(value) + "\n")

def write_note_count(file_name, title, value1, value2):
    write_dir = '../evaluation_results/overworld_theme/'
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    with open(write_dir + file_name, 'w') as f:
        f.write(title + "\n\n")
        f.write("Train_set Note count: "+ str(value1) + "\n")
        f.write("AI_val_set Note count: "+ str(value2) + "\n")


# Params
val_dir = "../evaluation_sets/overworld_theme"
thresh = 0.25

# Evaluation Train Set
print("evaluating Train_Set:")
train_set = np.load(val_dir + '/train_set_samples.npy')
print("Test_Set Shape: ", train_set.shape)
train_eb_rate = evaluate_eb(train_set)
train_dp_rate = evaluate_dp(train_set, thresh)
train_poly_rate = evaluate_polyphonicity(train_set, thresh)
train_upc_avg = evaluate_upc_average(train_set, thresh)
train_td_rate = evaluate_tonal_distance(train_set, thresh)

train_evaluation_values = [train_eb_rate,
                           train_dp_rate, train_poly_rate, train_upc_avg, train_td_rate]

write_evaluation('train_set_evaluation.txt', "Train_Results Evaluation",
                 metric_lables, train_evaluation_values)

# Evaluation AI-Result-set
print("evaluating AI_Results")
val_set = np.load(val_dir + '/testsamples.npy')
print("AI_Results Shape: ", val_set.shape)
val_eb_rate = evaluate_eb(val_set)
val_dp_rate = evaluate_dp(val_set, thresh)
val_poly_rate = evaluate_polyphonicity(val_set, thresh)
val_upc_avg = evaluate_upc_average(val_set, thresh)
val_td_rate = evaluate_tonal_distance(val_set, thresh)

val_evaluation_values = [val_eb_rate, val_dp_rate,
                         val_poly_rate, val_upc_avg, val_td_rate]

write_evaluation('aI_val_set_evaluation.txt',
                 "AI_Results Evaluation", metric_lables, val_evaluation_values

# val1= evaluate_notes_per_song(train_set, thresh)
# val2= evaluate_notes_per_song(val_set, thresh)

# write_note_count('note_counts.txt', "Note Counts:", val1, val2)
