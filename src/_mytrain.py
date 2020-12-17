import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Lambda, TimeDistributed, Reshape, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import *
from data import *

import sys
import random
import os
import numpy as np
from matplotlib import pyplot as plt
import pydot
import cv2
import util
import midi

import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

K.set_image_data_format('channels_first')
VAE_B1 = 0.02
VAE_B2 = 0.1
EPOCHS = 2000
BATCH_SIZE = 200
WRITE_HISTORY = True
NUM_RAND_SONGS = 50
PLAY_ONLY = False

if WRITE_HISTORY:
    # Create folder to save models into
    if not os.path.exists('../History'):
        os.makedirs('../History')

params = {
    "optimizer": Adam,
    "use_embedding": False,
    "param_size": 120,
    "activation_str": 'relu',
    "max_length": 16,
    "bn_m": 0.9,
    "do_rate": 0.1,
    "lr": 0.001
}

data = Data(use_embedding=False)
x_train, y_train, x_shape, y_shape = data.get_train_set()


def plotScores(scores, fname, on_top=True):
    plt.clf()
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.grid(True)
    plt.plot(scores)
    plt.ylim([0.0, 0.009])
    plt.xlabel('Epoch')
    loc = ('upper right' if on_top else 'lower right')
    plt.draw()
    plt.savefig(fname)


def save_config():
    with open('../model/config.txt', 'w') as fout:
        fout.write('LR:          ' + str(params["lr"]) + '\n')
        fout.write('BN_M:        ' + str(params["bn_m"]) + '\n')
        fout.write('BATCH_SIZE:  ' + str(BATCH_SIZE) + '\n')
        fout.write('NUM_OFFSETS: ' + str(data.num_offsets) + '\n')
        fout.write('DO_RATE:     ' + str(params["do_rate"]) + '\n')
        fout.write('num_songs:   ' + str(data.num_songs) + '\n')
        fout.write('optimizer:   ' + type(model.optimizer).__name__ + '\n')


z_log_sigma_sq = 0.0
z_mean = 0.0


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
    kl_loss = VAE_B2 * K.mean(1 + z_log_sigma_sq -
                              K.square(z_mean) - K.exp(z_log_sigma_sq), axis=None)
    return xent_loss - kl_loss


test_ix = 0
y_test_song = np.copy(y_train[test_ix:test_ix+1])
x_test_song = np.copy(x_train[test_ix:test_ix+1])
midi.samples_to_midi(y_test_song[0], '../model/gt.mid', 16)


def model_fn(optimizer, use_embedding, param_size, activation_str, max_length, bn_m, do_rate, lr):
    if use_embedding:
        x_in = Input(shape=x_shape[1:])
        x = Embedding(x_train.shape[0], param_size, input_length=1)(x_in)
        x = Flatten(name='pre_encoder')(x)
    else:
        x_in = Input(shape=y_shape[1:])
        x = Reshape((y_shape[1], -1))(x_in)

        x = TimeDistributed(Dense(2000, activation=activation_str))(x)

        x = TimeDistributed(Dense(200, activation=activation_str))(x)

        x = Flatten()(x)

        x = Dense(1600, activation=activation_str)(x)

        z_mean = Dense(param_size)(x)
        z_log_sigma_sq = Dense(param_size)(x)
        x = Lambda(vae_sampling, output_shape=(param_size,),
                   name='pre_encoder')([z_mean, z_log_sigma_sq])

        # x = Dense(param_size)(x)
        # x = BatchNormalization(momentum=bn_m, name='pre_encoder')(x)

    x = Dense(1600, name='encoder')(x)
    x = BatchNormalization(momentum=bn_m)(x)
    x = Activation(activation_str)(x)
    if do_rate > 0:
        x = Dropout(do_rate)(x)

    x = Dense(max_length * 200)(x)
    x = Reshape((max_length, 200))(x)
    x = TimeDistributed(BatchNormalization(momentum=bn_m))(x)
    x = Activation(activation_str)(x)
    if do_rate > 0:
        x = Dropout(do_rate)(x)

    x = TimeDistributed(Dense(2000))(x)
    x = TimeDistributed(BatchNormalization(momentum=bn_m))(x)
    x = Activation(activation_str)(x)
    if do_rate > 0:
        x = Dropout(do_rate)(x)

    x = TimeDistributed(Dense(y_shape[2] * y_shape[3], activation='sigmoid'))(x)
    x = Reshape((y_shape[1], y_shape[2], y_shape[3]))(x)

    model = Model(x_in, x)
    model.compile(optimizer=optimizer(lr=lr), loss=vae_loss)

    plot_model(model, to_file='../model/model.png', show_shapes=True)

    return model, z_log_sigma_sq, z_mean

if PLAY_ONLY:
    print("Loading Model...")
    model= load_model('../model/model.h5', custom_objects={'VAE_B1': VAE_B1, 'vae_loss': vae_loss})
else:
    model, z_log_sigma_sq, z_mean = model_fn(**params)

save_config()
train_loss = []
ofs = 0

func = K.function([model.get_layer('encoder').input], [model.layers[-1].output])
enc = Model(inputs=model.input, outputs=model.get_layer('pre_encoder').output)

rand_vecs = np.random.normal(0.0, 1.0, (NUM_RAND_SONGS, params["param_size"]))
np.save('../model/rand.npy', rand_vecs)


def make_rand_songs(write_dir, rand_vecs):
    for i in range(rand_vecs.shape[0]):
        x_rand = rand_vecs[i:i+1]
        y_song = func([x_rand, 0])[0]
        midi.samples_to_midi(y_song[0], write_dir +
                             'rand' + str(i) + '.mid', 16, 0.25)


def make_rand_songs_normalized(write_dir, rand_vecs):
    if params["use_embedding"]:
        x_enc = np.squeeze(enc.predict(data.x_orig))
    else:
        x_enc = np.squeeze(enc.predict(data.y_orig))

    x_mean = np.mean(x_enc, axis=0)
    x_stds = np.std(x_enc, axis=0)
    x_cov = np.cov((x_enc - x_mean).T)
    u, s, v = np.linalg.svd(x_cov)
    e = np.sqrt(s)

    print("Means: ", x_mean[:6])
    print("Evals: ", e[:6])

    np.save(write_dir + 'means.npy', x_mean)
    np.save(write_dir + 'stds.npy', x_stds)
    np.save(write_dir + 'evals.npy', e)
    np.save(write_dir + 'evecs.npy', v)

    x_vecs = x_mean + np.dot(rand_vecs * e, v)
    make_rand_songs(write_dir, x_vecs)

    title = ''
    if '/' in write_dir:
        title = 'Epoch: ' + write_dir.split('/')[-2][1:]

    plt.clf()
    e[::-1].sort()
    plt.title(title)
    plt.bar(np.arange(e.shape[0]), e, align='center')
    plt.draw()
    plt.savefig(write_dir + 'evals.png')

    plt.clf()
    plt.title(title)
    plt.bar(np.arange(e.shape[0]), x_mean, align='center')
    plt.draw()
    plt.savefig(write_dir + 'means.png')

    plt.clf()
    plt.title(title)
    plt.bar(np.arange(e.shape[0]), x_stds, align='center')
    plt.draw()
    plt.savefig(write_dir + 'stds.png')


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.ofs = 0
        super()

    def on_epoch_begin(self, epoch, logs=None):
        cur_ix = 0
        for i in range(data.num_songs):
            end_ix = cur_ix + data.y_lengths[i]
            for j in range(params["max_length"]):
                k = (j + self.ofs) % (end_ix - cur_ix)
                y_train[i, j] = data.y_samples[cur_ix + k]
            cur_ix = end_ix
        assert(end_ix == data.num_samples)
        self.ofs += 1

    def on_epoch_end(self, epoch, logs={}):
        if WRITE_HISTORY:
            plotScores(train_loss, '../History/HistoryBattleTheme/Scores.png', True)
        else:
            plotScores(train_loss, '../model/Scores.png', True)

        if epoch in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450] or (epoch % 100 == 0):
            write_dir = ''
            if WRITE_HISTORY:
                write_dir = '../History/HistoryBattleTheme/e' + str(epoch)
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)
                write_dir += '/'
                model.save('../History/HistoryBattleTheme/model.h5')
                model.summary()

            else:
                model.save('model.h5')
            print("Saved")

            y_song = model.predict(y_test_song, batch_size=BATCH_SIZE)[0]
            util.samples_to_pics(write_dir + 'test', y_song)
            midi.samples_to_midi(y_song, write_dir + 'test.mid', 16)

            make_rand_songs(write_dir, rand_vecs)


callback = CustomCallback()

if not PLAY_ONLY:
    print("train")
    model.fit(
        x=y_train,
        y=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[callback]
    )

    y_samples = np.load('../model/samples.npy')
    y_lengths = np.load('../model/lengths.npy')

else:
    if not os.path.exists('TestSamples'):
        os.makedirs('TestSamples')

    write_dir = 'TestSamples/'
    print("Generating Songs...")
    make_rand_songs_normalized('TestSamples/', rand_vecs)
    for i in range(20):
        x_test_song = data.y_train[i:i+1]
        print (x_test_song.shape)
        y_song = model.predict(x_test_song, batch_size=BATCH_SIZE)[0]
        midi.samples_to_midi(y_song, 'TestSamples/'+'gt' + str(i) + '.mid', 16)
    exit(0)


# for i in range(10):
#     print("test")
#     util.samples_to_pics('samples/' + 'test' + str(i) , y_train[i])
#     midi.samples_to_midi(y_train[i], 'samples/testtest' + str(i) +'.mid', 96)
# print(y_train[1])
