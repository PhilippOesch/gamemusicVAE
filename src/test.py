import os
import util
import numpy as np
import matplotlib.pyplot as plt

import params
import musicVAE
import metrics

samples= musicVAE.y_train[50]
hcdf_results= metrics.hcdf(samples= samples)

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
