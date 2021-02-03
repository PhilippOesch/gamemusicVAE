import os
import utility.util as util
import utility.midi as midi
import numpy as np
import matplotlib.pyplot as plt

import params
import network.musicVAE as musicVAE
import utility.metrics as metrics

# samples= musicVAE.y_train[70]
# samples2= musicVAE.y_train[80]

# dp_rate= metrics.drum_pattern(samples)
# print("Rate of notes in a Drum-Pattern:", dp_rate)

# qn, qn_notes, notes= metrics.get_qn(samples= samples)

# print("Notes: ", notes, ", QN_Notes: ", qn_notes, ", QN Value: ", qn)
# hcdf_results= metrics.hcdf(samples= samples)

# td_total, td_per_frame= metrics.tonal_distance(samples, samples2)
# print("Tonal Distance:")
# print("Total= ", td_total)
# print("Per Frame= ", td_per_frame)

# #plot hcdf_results
# plt.plot(hcdf_results)
# plt.ylabel('TCDF')
# plt.xlabel('Time step')
# plt.show()

# upcs, average_upcs= metrics.get_upc(samples= samples)
# pp= metrics.polyphonicity(samples, relative_to_notes=False)

# # test_set= np.zeros(
# #         (16, 96, 88), dtype=np.float32)

# empty_bars, bars= metrics.empty_bars(samples)
# print(upcs)
# print("Average used pitch classes: ", average_upcs)
# print( empty_bars, " of ", bars, " bars empty")
# print("Poliphonicity:", pp)

# test_train= np.load('../evaluation_sets/overworld_theme/train_set_samples.npy')
# midi.samples_to_midi(test_train[1], '../samples/testtesttesttest.mid', 96)

val_dir = "../evaluation_sets/overworld_theme"

train_set = np.load(val_dir + '/train_set_samples.npy')
val_set = np.load(val_dir + '/testsamples.npy')

hcdf1= metrics.hcdf(samples= train_set[0], ignore_treshhold=False, thresh=0.25)
hcdf2= metrics.hcdf(samples= val_set[0], ignore_treshhold=False, thresh=0.25)
#plot hcdf_results
# plt.plot(hcdf_results)
# plt.ylabel('TCDF')
# plt.xlabel('Time step')
# plt.show()

plt.figure(figsize=(16, 9))
plt.subplot(211)
plt.title("Train-set")
line1= plt.plot(hcdf1)
plt.ylabel('TCDF')
plt.xlabel('Time step')
plt.setp(line1, color='r', linewidth=2.0)

plt.subplot(212)
plt.title("AI-Val-set")
line2= plt.plot(hcdf2)
plt.ylabel('TCDF')
plt.xlabel('Time step')
plt.setp(line2, color='b', linewidth=2.0)

plt.savefig('../evaluation_results/overworld_theme/HCDF_Train_to_val_comparison.png')


