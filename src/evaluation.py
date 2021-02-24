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

metric_lables_no_tonal_distance = [
    "Empty Bars",
    "Drum Pattern",
    "Polyphonicity",
    "Average amount of different pitch classes",
    "average Song Note count"
]



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


result = {}
lowest_td_result= 1
for i in range(5):
    random_idxs = np.random.choice(train_set_model2.shape[0], 10)
    val_set= [ train_set_model2[random_idxs[i]] for i in range(random_idxs.shape[0])]
    val_set= np.array(val_set)
    td_val, lowest_td_val, thresh_counter= metrics.evaluate_tonal_distance(
        data_set1=train_set_model1, data_set2=val_set, thresh=0.25)
    if lowest_td_val< lowest_td_result:
        lowest_td_result= lowest_td_val
    result["overworld to battle, i: " + str(i)] = td_val
    write_val_to_train_comparrison("Overworld_to_battle_comparisson_train2.txt",
                                    "Comparrison of train_set of both models", result, lowest_td_result, thresh_counter)
    
for i in range(5):
    random_idxs = np.random.choice(train_set_model1.shape[0], 10)
    val_set= [ train_set_model1[random_idxs[i]] for i in range(random_idxs.shape[0])]
    val_set= np.array(val_set)
    td_val, lowest_td_val, thresh_counter= metrics.evaluate_tonal_distance(
        data_set1=train_set_model2, data_set2=val_set, thresh=0.25)
    if lowest_td_val< lowest_td_result:
        lowest_td_result= lowest_td_val
    result["battle to overworld, i: " + str(i)] = td_val
    write_val_to_train_comparrison("Overworld_to_battle_comparisson_train2.txt",
                                    "Comparrison of train_set of both models", result, lowest_td_result, thresh_counter)


result1 = {}
result2 = {}
lowest_td_result1= 1
lowest_td_result2= 1

# Comparrison Overworld-Model
for i in range(5):
    random_idxs = np.random.choice(val_set_model1.shape[0], 10)
    val_set= [ val_set_model1[random_idxs[i]] for i in range(random_idxs.shape[0])]
    val_set= np.array(val_set)
    td_val, lowest_td_val, thresh_counter = metrics.evaluate_tonal_distance(
        data_set1=train_set_model1, data_set2=val_set, thresh=0.25)
    if lowest_td_val< lowest_td_result1:
        lowest_td_result1= lowest_td_val
    result1["i: " + str(i)] = td_val
    write_val_to_train_comparrison("overworld_train_to_val_comp2.txt",
                                   "Comparrison of train_set and generated samples of overworld_theme", result1, lowest_td_result1, thresh_counter)

# #Comparrison BattleTheme-Model
for i in range(5):
    random_idxs = np.random.choice(val_set_model2.shape[0], 10)
    val_set= [ val_set_model2[random_idxs[i]] for i in range(random_idxs.shape[0])]
    val_set= np.array(val_set)
    td_val, lowest_td_val, thresh_counter = metrics.evaluate_tonal_distance(
        data_set1=train_set_model2, data_set2=val_set, thresh=0.25)
    if lowest_td_val< lowest_td_result2:
        lowest_td_result2= lowest_td_val
    result2["i: " + str(i)] = td_val
    write_val_to_train_comparrison("battle_train_to_val_comp2.txt",
                                   "Comparrison of train_set and generated samples of battle_theme", result2, lowest_td_result2, thresh_counter)

_,td_test= metrics.tonal_distance(val_set_model1[0], val_set_model1[0], ignore_treshhold=False, thresh= thresh);
print(td_test)

# Evaluation Train Set
print("evaluating Train_Set1:")
train_set_model1 = np.load(val_dir1 + '/train_set_samples.npy')
print("Test_Set Shape: ", train_set_model1.shape)
model1_train_eval_results = [metrics.evaluate_eb(train_set_model1), metrics.evaluate_dp(train_set_model1, thresh), metrics.evaluate_polyphonicity(train_set_model1, thresh), metrics.evaluate_upc_average(
    train_set_model1, thresh), metrics.evaluate_tonal_distance(data_set1= train_set_model1[:100], thresh= thresh), metrics.evaluate_notes_per_song(train_set_model1, thresh)]


write_evaluation('train_set_evaluation2.txt', model1name, "Train_Results Evaluation",
                 metric_lables, model1_train_eval_results)

# Evaluation AI-Result-set
print("evaluating AI_Results1")
val_set_model1 = np.load(val_dir1 + '/testsamples.npy')
print("AI_Results Shape: ", val_set_model1.shape)
model1_val_eval_results = [metrics.evaluate_eb(val_set_model1), metrics.evaluate_dp(val_set_model1, thresh), metrics.evaluate_polyphonicity(val_set_model1, thresh), metrics.evaluate_upc_average(
    val_set_model1, thresh), metrics.evaluate_tonal_distance(data_set1= val_set_model1, thresh= thresh), metrics.evaluate_notes_per_song(val_set_model1, thresh)]


write_evaluation('aI_val_set_evaluation2.txt', model1name,
                 "AI_Results Evaluation", metric_lables, model1_val_eval_results)


# Evaluation Train Set
print("evaluating Train_Set2:")
train_set_model2 = np.load(val_dir2 + '/train_set_samples.npy')
print("Test_Set Shape: ", train_set_model2.shape)
model2_train_eval_results = [metrics.evaluate_eb(train_set_model2), metrics.evaluate_dp(train_set_model2, thresh), metrics.evaluate_polyphonicity(train_set_model2, thresh), metrics.evaluate_upc_average(
    train_set_model2, thresh), metrics.evaluate_tonal_distance(data_set1= train_set_model2[:100], thresh= thresh), metrics.evaluate_notes_per_song(train_set_model2, thresh)]

write_evaluation('train_set_evaluation2.txt', model2name, "Train_Results Evaluation",
                 metric_lables, model2_train_eval_results)


# Evaluation AI-Result-set
print("evaluating AI_Results2")
val_set_model2 = np.load(val_dir2 + '/testsamples.npy')
print("AI_Results Shape: ", val_set_model2.shape)
model2_val_eval_results = [metrics.evaluate_eb(val_set_model2), metrics.evaluate_dp(val_set_model2, thresh), metrics.evaluate_polyphonicity(val_set_model2, thresh), metrics.evaluate_upc_average(
    val_set_model2, thresh), metrics.evaluate_tonal_distance(data_set1= val_set_model2, thresh= thresh), metrics.evaluate_notes_per_song(val_set_model2, thresh)]


write_evaluation('aI_val_set_evaluation2.txt', model2name,
                 "AI_Results Evaluation", metric_lables, model2_val_eval_results)
