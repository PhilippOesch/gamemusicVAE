import os
import utility.util as util
import numpy as np
import matplotlib.pyplot as plt

import params
import network.musicVAE as musicVAE
import utility.metrics as metrics

model1name = "overworld_theme"
model2name = "battle_theme"

metric_lables = [
    "Empty Bars",
    "Drum Pattern",
    "Polyphonicity",
    "Average amount of different pitch classes",
    "Tonal Distance between tracks",
    "average Song Note count"
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
                    value, compare_value, ignore_treshhold= False, thresh= thresh)
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
    total_note_counter = 0

    for song in data_set:
        total_note_counter += metrics.get_note_count(song, thresh)

    return total_note_counter / data_set.shape[0]


def write_evaluation(file_name, modelname, title, metric_labels, values):
    write_dir = '../evaluation_results/' + modelname + '/'
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    with open(write_dir + file_name, 'w') as f:
        f.write(title + "\n\n")

        for metric, value in zip(metric_labels, values):
            f.write(metric + ": " + str(value) + "\n")


# Params
val_dir1 = "../evaluation_sets/" + model1name
val_dir2 = "../evaluation_sets/" + model2name
thresh = 0.25

# Evaluation Train Set
print("evaluating Train_Set1:")
train_set_model1 = np.load(val_dir1 + '/train_set_samples.npy')
print("Test_Set Shape: ", train_set_model1.shape)
model1_train_eval_results = [evaluate_eb(train_set_model1), evaluate_dp(train_set_model1, thresh), evaluate_polyphonicity(train_set_model1, thresh), evaluate_upc_average(
    train_set_model1, thresh), evaluate_tonal_distance(train_set_model1[:100], thresh), evaluate_notes_per_song(train_set_model1, thresh)]


write_evaluation('train_set_evaluation.txt', model1name, "Train_Results Evaluation",
                 metric_lables, model1_train_eval_results)

# Evaluation AI-Result-set
print("evaluating AI_Results1")
val_set_model1 = np.load(val_dir1 + '/testsamples.npy')
print("AI_Results Shape: ", val_set_model1.shape)
model1_val_eval_results = [evaluate_eb(val_set_model1), evaluate_dp(val_set_model1, thresh), evaluate_polyphonicity(val_set_model1, thresh), evaluate_upc_average(
    val_set_model1, thresh), evaluate_tonal_distance(val_set_model1, thresh), evaluate_notes_per_song(val_set_model1, thresh)]


write_evaluation('aI_val_set_evaluation.txt', model1name,
                 "AI_Results Evaluation", metric_lables, model1_val_eval_results)


# Evaluation Train Set
print("evaluating Train_Set2:")
train_set_model2 = np.load(val_dir2 + '/train_set_samples.npy')
print("Test_Set Shape: ", train_set_model2.shape)
model2_train_eval_results = [evaluate_eb(train_set_model2), evaluate_dp(train_set_model2, thresh), evaluate_polyphonicity(train_set_model2, thresh), evaluate_upc_average(
    train_set_model2, thresh), evaluate_tonal_distance(train_set_model2[:100], thresh), evaluate_notes_per_song(train_set_model2, thresh)]

write_evaluation('train_set_evaluation.txt', model2name, "Train_Results Evaluation",
                 metric_lables, model2_train_eval_results)


# Evaluation AI-Result-set
print("evaluating AI_Results2")
val_set_model2 = np.load(val_dir2 + '/testsamples.npy')
print("AI_Results Shape: ", val_set_model2.shape)
model2_val_eval_results = [evaluate_eb(val_set_model2), evaluate_dp(val_set_model2, thresh), evaluate_polyphonicity(val_set_model2, thresh), evaluate_upc_average(
    val_set_model2, thresh), evaluate_tonal_distance(val_set_model2, thresh), evaluate_notes_per_song(val_set_model2, thresh)]


write_evaluation('aI_val_set_evaluation.txt', model2name,
                 "AI_Results Evaluation", metric_lables, model2_val_eval_results)
