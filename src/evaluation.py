import os
import utility.util as util
import numpy as np
import matplotlib.pyplot as plt

import params
import network.musicVAE as musicVAE
import utility.metrics as metrics
from utility.mode_collapse import*
import random

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

similarity_threshhold = 0.01


def evaluate_dp(data_set, thresh):
    print("Evaluating rate of drum pattern rhythms in tracks")
    dp_sum = 0
    for song in data_set:
        dp_sum += metrics.drum_pattern(song, thresh)

    return dp_sum / data_set.shape[0]


def evaluate_eb(data_set, thresh):
    print("Evaluating Rate of Empty Bars")
    eb_sum = 0
    for song in data_set:
        eb_value, _ = metrics.empty_bars(song, thresh)
        eb_sum += eb_value

    return eb_sum / (data_set.shape[0] * data_set.shape[1])


def evaluate_polyphonicity(data_set, thresh):
    print("Evaluating Polyphonicity of tracks")
    poly_sum = 0
    for song in data_set:
        poly_sum += metrics.polyphonicity(song, True, thresh)

    return poly_sum / data_set.shape[0]


def evaluate_tonal_distance(data_set1, data_set2=[], thresh=0.25):
    print("Evaluating Tonal Distance between tracks")
    missing_sets = data_set1.copy()
    print(missing_sets.shape)
    total_sum = 0
    comparison_counter = 0
    lowestTD = 1
    under_thresh_counter= 0
    try:
        for value in data_set1:
            if not missing_sets.shape[0] == 0:
                if len(data_set2) == 0:
                    for compare_value in missing_sets[1:]:
                        print("Comparison Counter: ", comparison_counter)
                        comparison_counter += 1
                        _, comparisson_td, under_thresh = metrics.tonal_distance(
                            value, compare_value, ignore_treshhold=False, thresh=thresh)
                        if comparisson_td < similarity_threshhold:
                            raise ModeCollapse
                        total_sum += comparisson_td
                        if comparisson_td < lowestTD:
                            lowestTD = comparisson_td
                        under_thresh_counter+= under_thresh
                    missing_sets = np.delete(missing_sets, value, axis=0)
                else:
                    for compare_value in data_set2:
                        print("Comparison Counter: ", comparison_counter)
                        comparison_counter += 1
                        _, comparisson_td, under_thresh = metrics.tonal_distance(
                            value, compare_value, ignore_treshhold=False, thresh=thresh)
                        print(comparisson_td)
                        if comparisson_td < similarity_threshhold:
                            raise ModeCollapse
                        total_sum += comparisson_td
                        if comparisson_td < lowestTD:
                            lowestTD = comparisson_td
                        under_thresh_counter+= under_thresh
        print(comparison_counter)
    except ModeCollapse:
        print("Mode Collapse Exception was thrown. The samples are too similar")

    return total_sum / comparison_counter, lowestTD, under_thresh_counter


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


def write_val_to_train_comparrison(file_name, title, dictionary, lowestTD, thresh_count):
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
        f.write("Timesteps where tonal_distance is under 0.01: " + str(thresh_count) + "\n")


# Params
val_dir1 = "../evaluation/evaluation_sets/" + model1name
val_dir2 = "../evaluation/evaluation_sets/" + model2name
thresh = 0.25

train_set_model1 = np.load(val_dir1 + '/train_set_samples.npy')
train_set_model2 = np.load(val_dir2 + '/train_set_samples.npy')

val_set_model1 = np.load(val_dir1 + '/testsamples.npy')
val_set_model2 = np.load(val_dir2 + '/testsamples.npy')


# result = {}
# lowest_td_result= 1
# for i in range(5):
#     random_idxs = np.random.choice(train_set_model2.shape[0], 10)
#     val_set= [ train_set_model2[random_idxs[i]] for i in range(random_idxs.shape[0])]
#     val_set= np.array(val_set)
#     td_val, lowest_td_val, thresh_counter= evaluate_tonal_distance(
#         data_set1=train_set_model1, data_set2=val_set, thresh=0.25)
#     if lowest_td_val< lowest_td_result:
#         lowest_td_result= lowest_td_val
#     result["overworld to battle, i: " + str(i)] = td_val
#     write_val_to_train_comparrison("Overworld_to_battle_comparisson_train2.txt",
#                                     "Comparrison of train_set of both models", result, lowest_td_result, thresh_counter)
    
# for i in range(5):
#     random_idxs = np.random.choice(train_set_model1.shape[0], 10)
#     val_set= [ train_set_model1[random_idxs[i]] for i in range(random_idxs.shape[0])]
#     val_set= np.array(val_set)
#     td_val, lowest_td_val, thresh_counter= evaluate_tonal_distance(
#         data_set1=train_set_model2, data_set2=val_set, thresh=0.25)
#     if lowest_td_val< lowest_td_result:
#         lowest_td_result= lowest_td_val
#     result["battle to overworld, i: " + str(i)] = td_val
#     write_val_to_train_comparrison("Overworld_to_battle_comparisson_train2.txt",
#                                     "Comparrison of train_set of both models", result, lowest_td_result, thresh_counter)


# result1 = {}
# result2 = {}
# lowest_td_result1= 1
# lowest_td_result2= 1

# # Comparrison Overworld-Model
# for i in range(5):
#     random_idxs = np.random.choice(val_set_model1.shape[0], 10)
#     val_set= [ val_set_model1[random_idxs[i]] for i in range(random_idxs.shape[0])]
#     val_set= np.array(val_set)
#     td_val, lowest_td_val, thresh_counter = evaluate_tonal_distance(
#         data_set1=train_set_model1, data_set2=val_set, thresh=0.25)
#     if lowest_td_val< lowest_td_result1:
#         lowest_td_result1= lowest_td_val
#     result1["i: " + str(i)] = td_val
#     write_val_to_train_comparrison("overworld_train_to_val_comp2.txt",
#                                    "Comparrison of train_set and generated samples of overworld_theme", result1, lowest_td_result1, thresh_counter)

# # #Comparrison BattleTheme-Model
# for i in range(5):
#     random_idxs = np.random.choice(val_set_model2.shape[0], 10)
#     val_set= [ val_set_model2[random_idxs[i]] for i in range(random_idxs.shape[0])]
#     val_set= np.array(val_set)
#     td_val, lowest_td_val, thresh_counter = evaluate_tonal_distance(
#         data_set1=train_set_model2, data_set2=val_set, thresh=0.25)
#     if lowest_td_val< lowest_td_result2:
#         lowest_td_result2= lowest_td_val
#     result2["i: " + str(i)] = td_val
#     write_val_to_train_comparrison("battle_train_to_val_comp2.txt",
#                                    "Comparrison of train_set and generated samples of battle_theme", result2, lowest_td_result2, thresh_counter)

# _,td_test= metrics.tonal_distance(val_set_model1[0], val_set_model1[0], ignore_treshhold=False, thresh= thresh);
# print(td_test)

# Evaluation Train Set
print("evaluating Train_Set1:")
train_set_model1 = np.load(val_dir1 + '/train_set_samples.npy')
print("Test_Set Shape: ", train_set_model1.shape)
model1_train_eval_results = [evaluate_eb(train_set_model1), evaluate_dp(train_set_model1, thresh), evaluate_polyphonicity(train_set_model1, thresh), evaluate_upc_average(
    train_set_model1, thresh), evaluate_tonal_distance(data_set1= train_set_model1[:100], thresh= thresh), evaluate_notes_per_song(train_set_model1, thresh)]


write_evaluation('train_set_evaluation2.txt', model1name, "Train_Results Evaluation",
                 metric_lables, model1_train_eval_results)

# Evaluation AI-Result-set
print("evaluating AI_Results1")
val_set_model1 = np.load(val_dir1 + '/testsamples.npy')
print("AI_Results Shape: ", val_set_model1.shape)
model1_val_eval_results = [evaluate_eb(val_set_model1), evaluate_dp(val_set_model1, thresh), evaluate_polyphonicity(val_set_model1, thresh), evaluate_upc_average(
    val_set_model1, thresh), evaluate_tonal_distance(data_set1= val_set_model1, thresh= thresh), evaluate_notes_per_song(val_set_model1, thresh)]


write_evaluation('aI_val_set_evaluation2.txt', model1name,
                 "AI_Results Evaluation", metric_lables, model1_val_eval_results)


# Evaluation Train Set
print("evaluating Train_Set2:")
train_set_model2 = np.load(val_dir2 + '/train_set_samples.npy')
print("Test_Set Shape: ", train_set_model2.shape)
model2_train_eval_results = [evaluate_eb(train_set_model2), evaluate_dp(train_set_model2, thresh), evaluate_polyphonicity(train_set_model2, thresh), evaluate_upc_average(
    train_set_model2, thresh), evaluate_tonal_distance(data_set1= train_set_model2[:100], thresh= thresh), evaluate_notes_per_song(train_set_model2, thresh)]

write_evaluation('train_set_evaluation2.txt', model2name, "Train_Results Evaluation",
                 metric_lables, model2_train_eval_results)


# Evaluation AI-Result-set
print("evaluating AI_Results2")
val_set_model2 = np.load(val_dir2 + '/testsamples.npy')
print("AI_Results Shape: ", val_set_model2.shape)
model2_val_eval_results = [evaluate_eb(val_set_model2), evaluate_dp(val_set_model2, thresh), evaluate_polyphonicity(val_set_model2, thresh), evaluate_upc_average(
    val_set_model2, thresh), evaluate_tonal_distance(data_set1= val_set_model2, thresh= thresh), evaluate_notes_per_song(val_set_model2, thresh)]


write_evaluation('aI_val_set_evaluation2.txt', model2name,
                 "AI_Results Evaluation", metric_lables, model2_val_eval_results)
