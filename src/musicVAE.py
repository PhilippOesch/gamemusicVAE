from matplotlib import use
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Lambda, TimeDistributed, Reshape, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.backend import learning_phase
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import *
from data import *
from tensorflow.keras.backend import learning_phase

from params import *

vae_b1 = VAEparams["vae_b1"]
vae_b2 = VAEparams["vae_b2"]
data = Data(use_embedding=False)
x_train, y_train, x_shape, y_shape = data.get_train_set()
z_log_sigma_sq = 0.0
z_mean = 0.0

def vae_loss(x, x_decoded_mean):
    xent_loss= binary_crossentropy(x, x_decoded_mean)
    kl_loss= vae_b2* K.mean(1+ z_log_sigma_sq- K.square(z_mean)- K.exp(z_log_sigma_sq), axis= None)
    return xent_loss - kl_loss

def vae_sampling(args):
    z_mean, z_log_sigma_sq = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=vae_b1)
    return z_mean + K.exp(z_log_sigma_sq * 0.5) * epsilon

def get_encoder(param_size, bn_m, activation, do_rate, max_length):
    x_in = Input(name='encoder', shape=(param_size,))
    x = Dense(1600)(x_in)
    x = BatchNormalization(momentum=bn_m)(x)
    x = Activation(activation)(x)
    if do_rate > 0:
        x = Dropout(do_rate)(x)

    x = Dense(max_length * 200)(x)
    x = Reshape((max_length, 200))(x)
    x = TimeDistributed(BatchNormalization(momentum=bn_m))(x)
    x = Activation(activation)(x)
    if do_rate > 0:
        x = Dropout(do_rate)(x)

    x = TimeDistributed(Dense(2000))(x)
    x = TimeDistributed(BatchNormalization(momentum=bn_m))(x)
    x = Activation(activation)(x)
    if do_rate > 0:
        x = Dropout(do_rate)(x)

    x = TimeDistributed(
        Dense(y_shape[2] * y_shape[3], activation='sigmoid'))(x)
    x = Reshape((y_shape[1], y_shape[2], y_shape[3]))(x)

    if learning_phase:
        model = Model(x_in, x)
        # model= K.function([x_in], [x])
    else:
        model = Model(x_in, x)
        plot_model(model, to_file='../model/encoder_model.png',
                   show_shapes=True)

    return model


def get_pre_encoder(use_embedding, param_size, activation, max_length):
    if use_embedding:
        x_in = Input(shape=x_shape[1:])
        x = Embedding(x_train.shape[0], param_size, input_length=1)(x_in)
        x= Flatten(name='pre_encoder')(x)
        model= Model(x_in, x)
    else:
        x_in= Input(shape= y_shape[1:])
        x= Reshape((y_shape[1], -1))(x_in)

        x = TimeDistributed(Dense(2000, activation= activation))(x)

        x = TimeDistributed(Dense(200, activation= activation))(x)

        x= Flatten()(x)

        x= Dense(1600, activation= activation)(x)

        z_mean= Dense(param_size)(x)
        z_log_sigma_sq= Dense(param_size)(x)
        x_out= Lambda(vae_sampling, output_shape=(param_size,), name= 'pre_encoder')([z_mean, z_log_sigma_sq])

        model= Model(x_in, x_out)

        plot_model(model, to_file='../model/pre_encoder_model.png', show_shapes=True)

        return model

def build_full_model(optimizer, use_embedding, param_size, activation_str, max_length, bn_m, do_rate, lr, vae_b1, vae_b2, epochs, batch_size, write_history, num_rand_songs, play_only, history_dir, log_dir, continue_training):
    pre_encoder = get_pre_encoder(use_embedding= use_embedding, param_size= param_size, activation= activation_str, max_length= max_length)
    encoder = get_encoder(param_size= param_size, bn_m= bn_m, activation=activation_str, do_rate=do_rate, max_length=max_length)

    x_in = pre_encoder.input
    x_out = encoder(pre_encoder.output)
    model = Model(x_in, x_out)
    model.compile(optimizer=optimizer(lr=lr), loss=vae_loss)

    plot_model(model, to_file='../model/model.png', show_shapes=True)

    return model, encoder, pre_encoder
