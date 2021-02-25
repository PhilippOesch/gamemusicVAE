
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import utility.midi as midi
import numpy as np
import pydot
import scipy
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.backend import learning_phase
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV, ParameterGrid
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import os
import utility.util as util

from params import VAEparams, GeneralParams
import network.musicVAE as musicVAE
tf.keras.backend.clear_session()


model= musicVAE.build_full_model(VAEparams["optimizer"], GeneralParams["vae_b2"], VAEparams["lr"])

i= 0
score= []

model_log_dir = os.path.join(
    GeneralParams["log_dir"], GeneralParams["model_name"])

tb_callback = TensorBoard(
    log_dir=model_log_dir,
    histogram_freq=1
)


#Custom Cross Val
model.fit(
    x=musicVAE.y_train,
    y=musicVAE.y_train,
    epochs=VAEparams["epochs"],
    batch_size=VAEparams["batch_size"],
    callbacks=[tb_callback],
)
score= model.get_metrics()
print(score)

with open("../parameter_tuning/" + GeneralParams["model_name"] + ".txt", 'w') as f:
    f.write("Model1" + "\n\n")
    for key in VAEparams:
        f.write(key + ": " + str(VAEparams[key]) + " - ")

    f.write("\nbeta: " + str(GeneralParams["vae_b2"]) + " - \n\n")

    f.write("Loss: "+ str(score[0])+ "\n")
    f.write("Reconstruction-Loss: "+ str(score[1])+ "\n")
    f.write("KL-Divergenz: "+ str(score[2])+ "\n\n")

    f.write(str(GeneralParams["model_name"]))
