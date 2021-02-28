import os
import utility.util as util
import numpy as np
import matplotlib.pyplot as plt

import params
import network.musicVAE as musicVAE
import utility.metrics as metrics
from utility.mode_collapse import*
import random

from params import GeneralParams

model1name = "battle_theme"
model2name = "overworld_theme"

# Params
val_dir1 = "../evaluation/evaluation_sets/" + model1name
val_dir2 = "../evaluation/evaluation_sets/" + model2name
thresh = GeneralParams["thresh"]

train_set_model1 = np.load(val_dir1 + '/train_set_samples.npy')
train_set_model2 = np.load(val_dir2 + '/train_set_samples.npy')

val_set_model1 = np.load(val_dir1 + '/battle_final_1_samples.npy')
val_set_model2 = np.load(val_dir2 + '/overworld_final_1_samples.npy')


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

#Comparrison BattleTheme-Model
for i in range(5):
    random_idxs = np.random.choice(val_set_model2.shape[0], 10)
    val_set= [ val_set_model2[random_idxs[i]] for i in range(random_idxs.shape[0])]
    val_set= np.array(val_set)
    td_val, lowest_td_val = metrics.evaluate_tonal_distance(
        data_set1=train_set_model2, data_set2=val_set, thresh=0.25)
    if lowest_td_val< lowest_td_result2:
        lowest_td_result2= lowest_td_val
    result2["i: " + str(i)] = td_val
    metrics.write_val_to_train_comparrison("battle_train_to_val_comp.txt",
                                   "Comparrison of train_set and generated samples of battle_theme", result2, lowest_td_result2)


# Evaluation Train Set
print("evaluating Train_Set1:")
train_set_model1 = np.load(val_dir1 + '/train_set_samples.npy')
print("Test_Set Shape: ", train_set_model1.shape)
metrics.write_td_evaluation("battle_train_set_td.txt" ,model1name, "Battle Train Set Evaluation:", metrics.evaluate_tonal_distance(data_set1= train_set_model1[:100], thresh= thresh))

# Evaluation AI-Result-set
print("evaluating AI_Results1")
val_set_model1 = np.load(val_dir1 + '/battle_final_1_samples.npy')
print("AI_Results Shape: ", val_set_model1.shape)
metrics.write_td_evaluation("battle_sample_set_td.txt" ,model1name, "Battle Sample Set Evaluation:", metrics.evaluate_tonal_distance(data_set1= val_set_model1, thresh= thresh))


# Evaluation Train Set
print("evaluating Train_Set2:")
train_set_model2 = np.load(val_dir2 + '/train_set_samples.npy')
print("Test_Set Shape: ", train_set_model2.shape)
metrics.write_td_evaluation("overworld_train_set_td.txt" ,model2name, "Overworld Train Set Evaluation:", metrics.evaluate_tonal_distance(data_set1= train_set_model2[:100], thresh= thresh))


# Evaluation AI-Result-set
print("evaluating AI_Results2")
val_set_model2 = np.load(val_dir2 + '/overworld_final_1_samples.npy')
print("AI_Results Shape: ", val_set_model2.shape)
metrics.write_td_evaluation("overworld_sample_set_td.txt" ,model2name, "Overworld Train Set Evaluation:", metrics.evaluate_tonal_distance(data_set1= val_set_model2, thresh= thresh))
