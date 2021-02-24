from typing import NoReturn
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
from tensorflow import keras
from tensorflow.keras import layers

from params import *

beta = GeneralParams["vae_b2"]
data = Data(GeneralParams["dataset_name"])
y_train, y_shape = data.get_train_set()


class Sampling(layers.Layer):

    def call(self, inputs):
        mu, sigma = inputs
        epsilon = K.random_normal(shape=K.shape(mu))
        return mu + tf.exp(0.5 * sigma) * epsilon


# encoder
encoder_in = Input(shape=y_shape[1:])
x = Reshape((y_shape[1], -1))(encoder_in)
print(x.shape)
x = TimeDistributed(
    Dense(VAEparams["dim1"], activation=VAEparams["activation_str"]))(x)
x = TimeDistributed(
    Dense(VAEparams["dim2"], activation=VAEparams["activation_str"]))(x)
x = Flatten()(x)
x = Dense(VAEparams["dim3"], activation=VAEparams["activation_str"])(x)
mu = Dense(VAEparams["param_size"])(x)
sigma = Dense(VAEparams["param_size"])(x)
z = Sampling()([mu, sigma])
encoder = Model(encoder_in, [mu, sigma, z])
encoder.summary()
plot_model(encoder, to_file='../model/encoder_model.png',
           show_shapes=True)

# decoder
decoder_in = Input(name='encoder', shape=(VAEparams["param_size"],))
x = Dense(VAEparams["dim3"])(decoder_in)
if VAEparams["use_batchnorm"]:
    x = BatchNormalization(momentum=VAEparams["bn_m"])(x)
x = Activation(VAEparams["activation_str"])(x)
if VAEparams["do_rate"] > 0:
    x = Dropout(VAEparams["do_rate"])(x)
x = Dense(VAEparams["max_length"] * VAEparams["dim2"])(x)
x = Reshape((VAEparams["max_length"], VAEparams["dim2"]))(x)
if VAEparams["use_batchnorm"]:
    x = TimeDistributed(BatchNormalization(momentum=VAEparams["bn_m"]))(x)
x = Activation(VAEparams["activation_str"])(x)
if VAEparams["do_rate"] > 0:
    x = Dropout(VAEparams["do_rate"])(x)
x = TimeDistributed(Dense(VAEparams["dim1"]))(x)
if VAEparams["use_batchnorm"]:
    x = TimeDistributed(BatchNormalization(momentum=VAEparams["bn_m"]))(x)
x = Activation(VAEparams["activation_str"])(x)
if VAEparams["do_rate"] > 0:
    x = Dropout(VAEparams["do_rate"])(x)
x = TimeDistributed(
    Dense(y_shape[2] * y_shape[3], activation='sigmoid'))(x)
x = Reshape((y_shape[1], y_shape[2], y_shape[3]))(x)
if learning_phase:
    decoder = Model(decoder_in, x)
    # model= K.function([x_in], [x])
else:
    decoder = Model(decoder_in, x)
decoder.summary()
plot_model(decoder, to_file='../model/encoder_model.png',
           show_shapes=True)


class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            mu, sigma, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # loss calculation
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2, 3)
                )
            )
            kl_loss = -0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + (self.beta * kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def get_metrics(self):
        return [
            self.total_loss_tracker.result().numpy(),
            self.reconstruction_loss_tracker.result().numpy(),
            self.kl_loss_tracker.result().numpy(),
        ]


def build_full_model(optimizer, beta, lr):
    vae = VAE(encoder, decoder, beta)
    vae.compile(optimizer=optimizer(lr=lr))
    return vae
