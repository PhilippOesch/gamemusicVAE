import os
import util
import numpy as np
import matplotlib.pyplot as plt

import params
import musicVAE
import metrics

def evaluate_dp(data_set, thresh):
    dp_sum= 0
    for song in data_set:
        dp_sum+= metrics.drum_pattern(song)

    return dp_sum/ data_set.shape[0]

def evaluate_eb(data_set):
    eb_sum= 0
    for song in data_set:
        eb_value, _ = metrics.empty_bars(song)
        eb_sum += eb_value
    
    return eb_sum/ (data_set.shape[0]* data_set.shape[1])

def evaluate_polyphonicity(data_set, thresh):
    poly_sum= 0
    for song in data_set:
        poly_sum+= metrics.polyphonicity(song, True, thresh)
    
    return poly_sum/ data_set.shape[0]

def evaluate_tonal_distance(data_set, thresh):
    missing_sets= data_set.copy()
    print(missing_sets.shape)
    total_sum= 0
    comparison_counter= 0
    for value in data_set:
        print(missing_sets.shape)
        if not missing_sets.shape[0] == 0:
            for compare_value in missing_sets[1:]:
                print("Comparison Counter: ",comparison_counter)
                comparison_counter+= 1
                _, comparisson_td= metrics.tonal_distance(value, compare_value)
                total_sum+= comparisson_td
            missing_sets= np.delete(missing_sets, value, axis= 0)
    print(comparison_counter)

    return total_sum/ comparison_counter


val_dir= "../evaluation_sets/overworld_theme"
val_set= np.load(val_dir+ '/testsamples.npy')
thresh= 0.5

# dp_rate= evaluate_dp(val_set, thresh)
# eb_rate= evaluate_eb(val_set)
# poly_rate= evaluate_polyphonicity(val_set, thresh)
td_rate= evaluate_tonal_distance(val_set, thresh)

# print("DP-Rate: ", dp_rate)
# print("EB-Rate: ", eb_rate)
# print("Poly-Rate: ", poly_rate)
print("TD-Rate: ", td_rate)