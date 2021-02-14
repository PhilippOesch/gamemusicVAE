import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Embedding, Flatten, Reshape, BatchNormalization, TimeDistributed, Dense, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.backend import learning_phase
from tensorflow.keras.utils import plot_model
from network.data import *

data = Data(use_embedding=False)
x_train, y_train, x_shape, y_shape = data.get_train_set()

class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class MusicVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(MusicVAE, self).__init__(**kwargs)
        self.encoder= encoder
        self.decoder= decoder
        self.total_loss_tracker= Mean(name="total_loss")
        self.reconstruction_loss_tracker= Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return[
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
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
        x= self.encoder(x)
        x= self.decoder(x)
        return x

def get_encoder(use_embedding, param_size, activation):
    if use_embedding:
        x_in= Input(shape= x_shape[1:])
        x = Embedding(x_train.shape[0], param_size, input_length=1)(x_in)
        x= Flatten(name='pre_encoder')(x)
        encoder= Model(x_in, x)
    else:
        x_in= Input(shape= y_shape[1:])
        x= Reshape((y_shape[1], -1))(x_in)
        x = TimeDistributed(Dense(2000, activation= activation))(x)
        x = TimeDistributed(Dense(200, activation= activation))(x)
        x= Flatten()(x)
        x= Dense(1600, activation= activation)(x)
        z_mean= Dense(param_size, name="z_mean")(x)
        z_log_var= Dense(param_size, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder= Model(x_in, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
    return encoder

def get_decoder(param_size, bn_m, activation, do_rate, max_length):
    x_in = Input(shape=(param_size,))
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

    decoder = Model(x_in, x, name="decoder")
    decoder.summary()
    return decoder
        
def build_full_model(optimizer, use_embedding, param_size, activation_str, max_length, bn_m, do_rate, lr, vae_b1, vae_b2, epochs, batch_size, write_history, num_rand_songs, play_only, history_dir, log_dir, continue_training, createTestingValues):
    encoder= get_encoder(use_embedding, param_size, activation_str)
    decoder= get_decoder(param_size, bn_m, activation_str, do_rate, max_length)
    vae= MusicVAE(encoder, decoder)
    vae.compile(optimizer= optimizer(lr=lr))

    plot_model(vae, to_file='../model/model.png', show_shapes=True)

    return vae, decoder, encoder