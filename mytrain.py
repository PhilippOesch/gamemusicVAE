from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import *
from tensorflow.keras.layers import *

import tensorflow as tf
import os, math
import sys, random, os
import numpy as np
from matplotlib import pyplot as plt
import pydot
import cv2
import util
import midi
from data import *


def to_song(encoded_output):
    	return np.squeeze(decoder([np.round(encoded_output), 0])[0])

def reg_mean_std(x):
	s = K.log(K.sum(x * x))
	return s*s

def vae_sampling(args):
	z_mean, z_log_sigma_sq = args
	epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=VAE_B1)
	return z_mean + K.exp(z_log_sigma_sq * 0.5) * epsilon

def vae_loss(x, x_decoded_mean):
	xent_loss = binary_crossentropy(x, x_decoded_mean)
	kl_loss = VAE_B2 * K.mean(1 + z_log_sigma_sq - K.square(z_mean) - K.exp(z_log_sigma_sq), axis=None)
	return xent_loss - kl_los

params = {
   "optimizer": Adam,
   "use_embedding": False,
   "param_size": 120,
   "activation_str": 'relu'
}

data= Data(use_embedding= False)
x_train, y_train, x_shape, y_shape= data.get_train_set()

def model_fn(optimizer, use_embedding, param_size, activation_str):
    if use_embedding:
        x_in= Input(shape= x_shape[1:])
        x = Embedding(x_train.shape[0], param_size, input_length=1)(x_in)
        x = Flatten('pre_encoder')
    else:
        x_in = Input(shape=y_shape[1:])
        x = Reshape((y_shape[1:]))(x_in)
        
        x = TimeDistributed(Dense(2000, activation= activation_str))

        x = TimeDistributed(Dense(200, activation= activation_str))

        x = Flatten()(x)

        x= Dense(1600, activation= activation_str)(x)

        z_mean =