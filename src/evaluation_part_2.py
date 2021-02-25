import os
import utility.util as util
import numpy as np
import matplotlib.pyplot as plt

import params
import network.musicVAE as musicVAE
import utility.metrics as metrics
from utility.mode_collapse import*
import random

model1name = "battle_theme"
model2name = "overworld_theme"

metric_lables = [
    "Empty Bars",
    "Drum Pattern",
    "Polyphonicity",
    "Average amount of different pitch classes",
    "Tonal Distance between tracks",
    "average Song Note count"
]


# Params
val_dir1 = "../evaluation/evaluation_sets/" + model1name
val_dir2 = "../evaluation/evaluation_sets/" + model2name
thresh = 0.25

train_set_model1 = np.load(val_dir1 + '/train_set_samples.npy')
train_set_model2 = np.load(val_dir2 + '/train_set_samples.npy')

val_set_model1 = np.load(val_dir1 + '/battle_final_3_samples.npy')
val_set_model2 = np.load(val_dir2 + '/overworld_final_3_samples.npy')


result = {}
lowest_td_result= 1
for i in range(5):
    random_idxs = np.random.choice(train_set_model2.shape[0], 10)
    val_set= [ train_set_model2[random_idxs[i]] for i in range(random_idxs.shape[0])]
    val_set= np.array(val_set)
    td_val, lowest_td_val= metrics.evaluate_tonal_distance(
        data_set1=train_set_model1, data_set2=val_set, thresh=0.25)
    if lowest_td_val< lowest_td_result:
        lowest_td_result= lowest_td_val
    result["overworld to battle, i: " + str(i)] = td_val
    metrics.write_val_to_train_comparrison("Overworld_to_battle_comparisson_train.txt",
                                    "Comparrison of train_set of both models", result, lowest_td_result)
    
for i in range(5):
    random_idxs = np.random.choice(train_set_model1.shape[0], 10)
    val_set= [ train_set_model1[random_idxs[i]] for i in range(random_idxs.shape[0])]
    val_set= np.array(val_set)
    td_val, lowest_td_val= metrics.evaluate_tonal_distance(
        data_set1=train_set_model2, data_set2=val_set, thresh=0.25)
    if lowest_td_val< lowest_td_result:
        lowest_td_result= lowest_td_val
    result["battle to overworld, i: " + str(i)] = td_val
    metrics.write_val_to_train_comparrison("Overworld_to_battle_comparisson_train.txt",
                                    "Comparrison of train_set of both models", result, lowest_td_result)


result1 = {}
result2 = {}
lowest_td_result1= 1
lowest_td_result2= 1

# Comparrison Overworld-Model
for i in range(5):
    random_idxs = np.random.choice(val_set_model1.shape[0], 10)
    val_set= [ val_set_model1[random_idxs[i]] for i in range(random_idxs.shape[0])]
    val_set= np.array(val_set)
    td_val, lowest_td_val = metrics.evaluate_tonal_distance(
        data_set1=train_set_model1, data_set2=val_set, thresh=0.25)
    if lowest_td_val< lowest_td_result1:
        lowest_td_result1= lowest_td_val
    result1["i: " + str(i)] = td_val
    metrics.write_val_to_train_comparrison("overworld_train_to_val_comp.txt",
                                   "Comparrison of train_set and generated samples of overworld_theme", result1, lowest_td_result1)

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
    metrics.write_val_to_train_comparrison("battle_train_to_val_comp.txt",
                                   "Comparrison of train_set and generated samples of battle_theme", result2, lowest_td_result2, thresh_counter)

_,td_test= metrics.tonal_distance(val_set_model1[0], val_set_model1[0], ignore_treshhold=False, thresh= thresh);
print(td_test)

# Evaluation Train Set
print("evaluating Train_Set1:")
train_set_model1 = np.load(val_dir1 + '/train_set_samples.npy')
random_idxs = np.random.choice(train_set_model1.shape[0], 100)
train_set_model1_reduced= [ train_set_model1[random_idxs[i]] for i in range(random_idxs.shape[0])]
train_set_model1_reduced= np.array(train_set_model1_reduced)
print("Test_Set Shape: ", train_set_model1.shape)
model1_train_eval_results = [metrics.evaluate_eb(train_set_model1), metrics.evaluate_dp(train_set_model1, thresh), metrics.evaluate_polyphonicity(train_set_model1, thresh), metrics.evaluate_upc_average(
    train_set_model1, thresh), metrics.evaluate_tonal_distance(data_set1= train_set_model1_reduced, thresh= thresh), metrics.evaluate_notes_per_song(train_set_model1, thresh)]


metrics.write_evaluation('train_set_battle_eval_part2.txt', model1name, "Train_Results Evaluation",
                 metric_lables, model1_train_eval_results)

# Evaluation AI-Result-set
print("evaluating AI_Results1")
val_set_model1 = np.load(val_dir1 + '/testsamples.npy')
print("AI_Results Shape: ", val_set_model1.shape)
model1_val_eval_results = [metrics.evaluate_eb(val_set_model1), metrics.evaluate_dp(val_set_model1, thresh), metrics.evaluate_polyphonicity(val_set_model1, thresh), metrics.evaluate_upc_average(
    val_set_model1, thresh), metrics.evaluate_tonal_distance(data_set1= val_set_model1, thresh= thresh), metrics.evaluate_notes_per_song(val_set_model1, thresh)]


metrics.write_evaluation('aI_set_eval_battle_part2.txt', model1name,
                 "AI_Results Evaluation", metric_lables, model1_val_eval_results)


# Evaluation Train Set
print("evaluating Train_Set2:")
train_set_model2 = np.load(val_dir2 + '/train_set_samples.npy')
random_idxs = np.random.choice(train_set_model2.shape[0], 100)
train_set_model2_reduced= [ train_set_model2[random_idxs[i]] for i in range(random_idxs.shape[0])]
train_set_model2_reduced= np.array(train_set_model2_reduced)
print("Test_Set Shape: ", train_set_model2.shape)
model2_train_eval_results = [metrics.evaluate_eb(train_set_model2), metrics.evaluate_dp(train_set_model2, thresh), metrics.evaluate_polyphonicity(train_set_model2, thresh), metrics.evaluate_upc_average(
    train_set_model2, thresh), metrics.evaluate_tonal_distance(data_set1= train_set_model2_reduced, thresh= thresh), metrics.evaluate_notes_per_song(train_set_model2, thresh)]

metrics.write_evaluation('train_set_overworld_eval_part2.txt', model2name, "Train_Results Evaluation",
                 metric_lables, model2_train_eval_results)


# Evaluation AI-Result-set
print("evaluating AI_Results2")
val_set_model2 = np.load(val_dir2 + '/testsamples.npy')
print("AI_Results Shape: ", val_set_model2.shape)
model2_val_eval_results = [metrics.evaluate_eb(val_set_model2), metrics.evaluate_dp(val_set_model2, thresh), metrics.evaluate_polyphonicity(val_set_model2, thresh), metrics.evaluate_upc_average(
    val_set_model2, thresh), metrics.evaluate_tonal_distance(data_set1= val_set_model2, thresh= thresh), metrics.evaluate_notes_per_song(val_set_model2, thresh)]


metrics.write_evaluation('aI_set_eval_overworld_part2.txt', model2name,
                 "AI_Results Evaluation", metric_lables, model2_val_eval_results)
