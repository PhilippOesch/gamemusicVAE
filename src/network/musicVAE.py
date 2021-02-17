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
from network.data import *
from tensorflow.keras.backend import learning_phase

from params import *

vae_b1 = GeneralParams["vae_b1"]
beta = GeneralParams["vae_b2"]
data = Data(use_embedding=False)
x_train, y_train, x_shape, y_shape = data.get_train_set()
sigma = 0.0
mu = 0.0


def vae_loss(x, x_decoded_mean):
    rec_loss = binary_crossentropy(x, x_decoded_mean)
    kld_loss =  -0.5 * K.mean(1 + sigma -
                              K.square(mu) - K.exp(sigma), axis=None)
    return rec_loss + (beta * kld_loss)


def vae_sampling(args):
    mu, sigma = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=vae_b1)
    return mu + K.exp(sigma * 0.5) * epsilon


def get_decoder(param_size, bn_m, activation, do_rate, max_length, use_batchnorm):
    x_in = Input(name='encoder', shape=(param_size,))
    x = Dense(1600)(x_in)
    if use_batchnorm:
        x = BatchNormalization(momentum=bn_m)(x)
    x = Activation(activation)(x)
    if do_rate > 0:
        x = Dropout(do_rate)(x)

    x = Dense(max_length * 200)(x)
    x = Reshape((max_length, 200))(x)
    if use_batchnorm:
        x = TimeDistributed(BatchNormalization(momentum=bn_m))(x)
    x = Activation(activation)(x)
    if do_rate > 0:
        x = Dropout(do_rate)(x)

    x = TimeDistributed(Dense(2000))(x)
    if use_batchnorm:
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

    plot_model(model, to_file='../model/decoder_model.png',
               show_shapes=True)


    return model


def get_encoder(param_size, activation, ignore_encoder_layer2):
    x_in = Input(shape=y_shape[1:])
    x = Reshape((y_shape[1], -1))(x_in)
    print(x.shape)

    x = TimeDistributed(Dense(2000, activation=activation))(x)

    if not ignore_encoder_layer2:
        x = TimeDistributed(Dense(200, activation=activation))(x)

    x = Flatten()(x)

    x = Dense(1600, activation=activation)(x)

    mu = Dense(param_size)(x)
    sigma = Dense(param_size)(x)
    z= Lambda(vae_sampling, output_shape=(param_size,),
                   name='pre_encoder')([mu, sigma])

    model = Model(x_in, [mu, sigma, z])

    plot_model(model, to_file='../model/encoder_model.png',
               show_shapes=True)

    return model


def build_full_model(optimizer, param_size, activation_str, max_length, bn_m, do_rate, lr, epochs, batch_size, use_batchnorm, ignore_encoder_layer2):
    encoder = get_encoder(
        param_size=param_size,activation=activation_str, ignore_encoder_layer2=ignore_encoder_layer2)
    decoder = get_decoder(param_size=param_size, bn_m=bn_m,
                          activation=activation_str, do_rate=do_rate, max_length=max_length, use_batchnorm= use_batchnorm)

    x_in = encoder.input
    x_out = decoder(encoder.output[2])
    model = Model(x_in, x_out)
    model.compile(optimizer=optimizer(lr=lr), loss=vae_loss)

    plot_model(model, to_file='../model/model.png', show_shapes=True)

    return model, decoder, encoder
