import os
import util
import numpy as np
import matplotlib.pyplot as plt

import params
import musicVAE
import metrics

samples= musicVAE.y_train[70]
samples2= musicVAE.y_train[80]

dp_rate= metrics.drum_pattern(samples)
print("Rate of notes in a Drum-Pattern:", dp_rate)

qn, qn_notes, notes= metrics.get_qn(samples= samples)

print("Notes: ", notes, ", QN_Notes: ", qn_notes, ", QN Value: ", qn)
hcdf_results= metrics.hcdf(samples= samples)

td_total, td_per_frame= metrics.tonal_distance(samples, samples2)
print("Tonal Distance:")
print("Total= ", td_total)
print("Per Frame= ", td_per_frame)

#plot hcdf_results
plt.plot(hcdf_results)
plt.ylabel('TCDF')
plt.xlabel('Time step')
plt.show()

upcs, average_upcs= metrics.get_upc(samples= samples)
pp= metrics.polyphonicity(samples, relative_to_notes=False)

# test_set= np.zeros(
#         (16, 96, 88), dtype=np.float32)

empty_bars, bars= metrics.empty_bars(samples)
print(upcs)
print("Average used pitch classes: ", average_upcs)
print( empty_bars, " of ", bars, " bars empty")
print("Poliphonicity:", pp)
