from tensorflow.keras.optimizers import Adam
from musicVAE import *
import midi
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
import os
import util

import params
import musicVAE

# Create folder to save models into
if not os.path.exists('../History'):
    os.makedirs('../History')

if not os.path.exists(VAEparams["history_dir"]):
    os.makedirs(VAEparams["history_dir"])

if not os.path.exists('../model'):
    os.makedirs('../model')

if not os.path.exists('../samples'):
    os.makedirs('../samples')

if not os.path.exists('../Testsamples'):
    os.makedirs('../Testsamples')

z_log_sigma_sq = 0.0
z_mean = 0.0

model, encoder, pre_encoder = musicVAE.build_full_model(**VAEparams)
if VAEparams["play_only"] or VAEparams["continue_training"] or VAEparams["createTestingValues"]:
    # encoder= load_model('../model/encoder_model')
    # pre_encoder= load_model('../model/pre_encoder_model')
    if VAEparams["write_history"]:
        model.load_weights(VAEparams["history_dir"]+'model_weights.h5')
    else:
        model.load_weights('../model/model_weights.h5')
    # pre_encoder= load_model('../model/pre_encoder_model.h5', custom_objects={'VAE_B1': musicVAE.vae_b1, 'vae_loss': musicVAE.vae_loss})

# save_config()
train_loss = []
ofs = 0

test_ix = 0
y_test_song = np.copy(musicVAE.y_train[test_ix:test_ix+1])
x_test_song = np.copy(musicVAE.x_train[test_ix:test_ix+1])
midi.samples_to_midi(y_test_song[0], '../model/gt.mid', 16)

# train

rand_vecs = np.random.normal(
    0.0, 1.0, (VAEparams["num_rand_songs"], VAEparams["param_size"]))
np.save('../model/rand.npy', rand_vecs)

# def to_song(encoded_output):
#     return np.squeeze(decoder([np.round(encoded_output), 0])[0])


def reg_mean_std(x):
    s = K.log(K.sum(x * x))
    return s*s

def make_rand_songs_and_get_result(write_dir, rand_vecs, thresh= 0.25):
    songs= []
    for i in range(rand_vecs.shape[0]):
        x_rand = rand_vecs[i:i+1]
        # print(x_rand.shape)
        y_song = encoder.predict(x_rand)
        songs.append(y_song[0])
        # y_song = func([x_rand, 0])[0]
        print(y_song.shape)
        # y_song = func([x_rand, 0])[0]
        midi.samples_to_midi(y_song[0], write_dir +
                             'rand' + str(i) + '.mid', 16, thresh)
    return np.array(songs)

def make_rand_songs(write_dir, rand_vecs, thresh= 0.25):
    for i in range(rand_vecs.shape[0]):
        x_rand = rand_vecs[i:i+1]
        # print(x_rand.shape)
        y_song = encoder.predict(x_rand)
        # y_song = func([x_rand, 0])[0]
        print(y_song.shape)
        # y_song = func([x_rand, 0])[0]
        midi.samples_to_midi(y_song[0], write_dir +
                             'rand' + str(i) + '.mid', 16, thresh)


def make_rand_songs_normalized(write_dir, rand_vecs, thresh= 0.25, getResult= False):
    if VAEparams["use_embedding"]:
        x_enc = np.squeeze(pre_encoder.predict(musicVAE.data.x_orig))
    else:
        x_enc = np.squeeze(pre_encoder.predict(musicVAE.data.y_orig))

    x_mean = np.mean(x_enc, axis=0)
    x_stds = np.std(x_enc, axis=0)
    x_cov = np.cov((x_enc - x_mean).T)
    print(x_cov.shape)
    # u, s, v = np.linalg.svd(x_cov)
    x_cov = np.nan_to_num(x_cov)
    u, s, v = scipy.linalg.svd(x_cov)
    e = np.sqrt(s)

    print("Means: ", x_mean[:6])
    print("Evals: ", e[:6])

    np.save(write_dir + 'means.npy', x_mean)
    np.save(write_dir + 'stds.npy', x_stds)
    np.save(write_dir + 'evals.npy', e)
    np.save(write_dir + 'evecs.npy', v)

    x_vecs = x_mean + np.dot(rand_vecs * e, v)
    if getResult:
        songs= make_rand_songs_and_get_result(write_dir, x_vecs, thresh)
    else:
        make_rand_songs(write_dir, x_vecs, thresh)

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

    if getResult:
        return songs
    else:
        return None

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


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.ofs = 0
        super()

    def on_epoch_begin(self, epoch, logs=None):
        cur_ix = 0
        for i in range(musicVAE.data.num_songs):
            end_ix = cur_ix + musicVAE.data.y_lengths[i]
            for j in range(VAEparams["max_length"]):
                k = (j + self.ofs) % (end_ix - cur_ix)
                musicVAE.y_train[i, j] = musicVAE.data.y_samples[cur_ix + k]
            cur_ix = end_ix
        assert(end_ix == musicVAE.data.num_samples)
        self.ofs += 1

    def on_epoch_end(self, epoch, logs={}):
        if VAEparams["write_history"]:
            plotScores(train_loss, VAEparams["history_dir"]+'Scores.png', True)
        else:
            plotScores(train_loss, '../model/Scores.png', True)

        if epoch in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450] or (epoch % 100 == 0):
            write_dir = ''
            if VAEparams["write_history"]:
                write_dir = VAEparams["history_dir"] + 'e' + str(epoch)
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)
                write_dir += '/'
                # model.save('../History/BattleTheme/model')
                # encoder.save('../History/BattleTheme/encoder_model')
                # pre_encoder.save('../History/BattleTheme/pre_encoder_model')
                model.save_weights(
                    VAEparams["history_dir"] + 'model_weights.h5')
                encoder.save_weights(
                    VAEparams["history_dir"] + 'encoder_weights.h5')
                pre_encoder.save_weights(
                    VAEparams["history_dir"] + 'pre_encoder_weights.h5')

                model.summary()

            else:
                model.save('../model/model.h5')
            print("Saved")

            y_song = model.predict(
                y_test_song, batch_size=VAEparams["batch_size"])[0]
            util.samples_to_pics(write_dir + 'test', y_song)
            midi.samples_to_midi(y_song, write_dir + 'test.mid', 16)

            make_rand_songs_normalized(write_dir, rand_vecs, 0.25)


callback = CustomCallback()

if VAEparams["play_only"]:
    # encoder= load_model('../model/encoder_model.h5')
    # # pre_encoder= load_model('../model/pre_encoder_model.h5', custom_objects={'VAE_B1': VAEparams["vae_b1"], 'vae_loss': vae_loss})
    write_dir= '../Testsamples/overworld_theme/';
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)


    make_rand_songs_normalized(write_dir + '/', rand_vecs, 0.6)
elif VAEparams["createTestingValues"]:
    write_dir= '../evaluation_samples/overworld_theme';
    testresults= make_rand_songs_normalized(write_dir + '/', rand_vecs, 0.25, True)

    np.save('../evaluation_sets/overworld_theme/testsamples.npy', testresults)
else:
    print("train")
    model.fit(
        x=musicVAE.y_train,
        y=musicVAE.y_train,
        epochs=VAEparams["epochs"],
        batch_size=VAEparams["batch_size"],
        callbacks=[callback],
    )

# for i in range(10):
#     print("test")
#     util.samples_to_pics('../samples/' + 'test' + str(i) , musicVAE.y_train[i])
#     midi.samples_to_midi(musicVAE.y_train[i], '../samples/testtest' + str(i) +'.mid', 96)

# model_log_dir = os.path.join(VAEparams["log_dir"], "model6")

# tb_callback = TensorBoard(
#     log_dir=model_log_dir,
#     histogram_freq=1,
#     write_graph=True
# )

# model.fit(
#     x=musicVAE.y_train,
#     y=musicVAE.y_train,
#     epochs=VAEparams["epochs"],
#     batch_size=VAEparams["batch_size"],
#     callbacks=[tb_callback],
# )

# scores = model.evaluate(
#     x=musicVAE.y_train,
#     y=musicVAE.y_train,
#     verbose=0
# )
# print(f"Scores before saving: {scores}")
