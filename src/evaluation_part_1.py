import os
import utility.util as util
import numpy as np
import matplotlib.pyplot as plt

import params
import utility.metrics as metrics
from utility.mode_collapse import*
import random

from params import VAEparams, GeneralParams

model1name = "battle_theme"
model2name = "overworld_theme"

metric_lables = [
    "Empty Bars",
    "Drum Pattern",
    "Polyphonicity",
    "Average amount of different pitch classes",
    "average Song Note count"
]



# Params
battle_val_dir = "../evaluation/evaluation_sets/" + model1name
overworld_val_dir = "../evaluation/evaluation_sets/" + model2name
thresh = GeneralParams["thresh"]

train_set_model1 = np.load(battle_val_dir + '/train_set_samples.npy')
train_set_model2 = np.load(overworld_val_dir + '/train_set_samples.npy')

battle1_val_set_model = np.load(battle_val_dir + '/battle_final_1_samples.npy')
print(battle1_val_set_model.shape)
battle2_val_set_model = np.load(battle_val_dir + '/battle_final_2_samples.npy')
overworld1_val_set_model = np.load(overworld_val_dir + '/overworld_final_1_samples.npy')
overworld2_val_set_model = np.load(overworld_val_dir + '/overworld_final_2_samples.npy')


#Train Battle
result_battle_train= [metrics.evaluate_eb(train_set_model1, thresh), metrics.evaluate_dp(train_set_model1, thresh), metrics.evaluate_polyphonicity(train_set_model1, thresh), metrics.evaluate_upc_average(
    train_set_model1, thresh), metrics.evaluate_notes_per_song(train_set_model1, thresh)]
metrics.write_evaluation('train_set_battle_evaluation_part1.txt', model1name, "Train_Results Evaluation",
                 metric_lables, result_battle_train)

#Train Overworld
result_overworld_train= [metrics.evaluate_eb(train_set_model2, thresh), metrics.evaluate_dp(train_set_model2, thresh), metrics.evaluate_polyphonicity(train_set_model2, thresh), metrics.evaluate_upc_average(
    train_set_model2, thresh), metrics.evaluate_notes_per_song(train_set_model2, thresh)]
metrics.write_evaluation('train_set_overworld_evaluation_part1.txt', model2name, "Train_Results Evaluation",
                 metric_lables, result_overworld_train)

#samples Battle 1
result_battle1= [metrics.evaluate_eb(battle1_val_set_model, thresh), metrics.evaluate_dp(battle1_val_set_model, thresh), metrics.evaluate_polyphonicity(battle1_val_set_model, thresh), metrics.evaluate_upc_average(
    battle1_val_set_model, thresh), metrics.evaluate_notes_per_song(battle1_val_set_model, thresh)]
metrics.write_evaluation('battle1_evaluation_part1.txt', model1name, "Train_Results Evaluation",
                 metric_lables, result_battle1)

#samples Battle 2
result_battle2= [metrics.evaluate_eb(battle2_val_set_model, thresh), metrics.evaluate_dp(battle2_val_set_model, thresh), metrics.evaluate_polyphonicity(battle2_val_set_model, thresh), metrics.evaluate_upc_average(
    battle2_val_set_model, thresh), metrics.evaluate_notes_per_song(battle2_val_set_model, thresh)]
metrics.write_evaluation('battle2_evaluation_part1.txt', model1name, "Train_Results Evaluation",
                 metric_lables, result_battle2)

#samples Overworld 1
result_overworld1= [metrics.evaluate_eb(overworld1_val_set_model, thresh), metrics.evaluate_dp(overworld1_val_set_model, thresh), metrics.evaluate_polyphonicity(overworld1_val_set_model, thresh), metrics.evaluate_upc_average(
    overworld1_val_set_model, thresh), metrics.evaluate_notes_per_song(overworld1_val_set_model, thresh)]
metrics.write_evaluation('overworld1_evaluation_part1.txt', model2name, "Train_Results Evaluation",
                 metric_lables, result_overworld1)

#samples Overworld 2
result_overworld2= [metrics.evaluate_eb(overworld2_val_set_model, thresh), metrics.evaluate_dp(overworld2_val_set_model, thresh), metrics.evaluate_polyphonicity(overworld2_val_set_model, thresh), metrics.evaluate_upc_average(
    overworld2_val_set_model, thresh), metrics.evaluate_notes_per_song(overworld2_val_set_model, thresh)]
metrics.write_evaluation('overworld2_evaluation_part1.txt', model2name, "Train_Results Evaluation",
                 metric_lables, result_overworld2)



