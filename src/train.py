from tensorflow.keras.optimizers import Adam
from musicVAE import *
import midi
import numpy as np
import pydot
import scipy
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.backend import learning_phase
from tensorflow.keras.models import Model, Sequential, load_model
import os
import util

params = {
    "optimizer": Adam,
    "use_embedding": False,
    "param_size": 120,
    "activation_str": 'relu',
    "max_length": 16,
    "bn_m": 0.9,
    "do_rate": 0.1,
    "lr": 0.001,
    "vae_b1": 0.02,
    "vae_b2": 0.1,
    "epochs": 2000,
    "batch_size": 200,
    "write_history": True,
    "num_rand_songs": 10,
    "play_only": False
}


# Create folder to save models into
if not os.path.exists('../History'):
    os.makedirs('../History')

if not os.path.exists('../History/BattleTheme'):
    os.makedirs('../History/BattleTheme')

if not os.path.exists('../model'):
    os.makedirs('../model')

musicVAE= MusicVAE(**params)
musicVAE.build_full_model()
model= musicVAE.model

# save_config()
train_loss = []
ofs = 0

test_ix = 0
y_test_song = np.copy(musicVAE.y_train[test_ix:test_ix+1])
x_test_song = np.copy(musicVAE.x_train[test_ix:test_ix+1])
midi.samples_to_midi(y_test_song[0], '../model/gt.mid', 16)

# train

encoder = musicVAE.encoder
pre_encoder = musicVAE.pre_encoder

rand_vecs = np.random.normal(0.0, 1.0, (params["num_rand_songs"], params["param_size"]))
np.save('../model/rand.npy', rand_vecs)

def to_song(encoded_output):
    return np.squeeze(decoder([np.round(encoded_output), 0])[0])


def reg_mean_std(x):
    s = K.log(K.sum(x * x))
    return s*s


def make_rand_songs(write_dir, rand_vecs):
    for i in range(rand_vecs.shape[0]):
        x_rand = rand_vecs[i:i+1]
        # print(x_rand.shape)
        y_song= encoder.predict(x_rand)
        # y_song = func([x_rand, 0])[0]
        print(y_song.shape)
        # y_song = func([x_rand, 0])[0]
        midi.samples_to_midi(y_song[0], write_dir +'rand' + str(i) + '.mid', 16, 0.25)


def make_rand_songs_normalized(write_dir, rand_vecs):
    if params["use_embedding"]:
        x_enc = np.squeeze(pre_encoder.predict(musicVAE.data.x_orig))
    else:
        x_enc = np.squeeze(pre_encoder.predict(musicVAE.data.y_orig))

    x_mean = np.mean(x_enc, axis=0)
    x_stds = np.std(x_enc, axis=0)
    x_cov = np.cov((x_enc - x_mean).T)
    print(x_cov.shape)
    # u, s, v = np.linalg.svd(x_cov)
    u, s, v = scipy.linalg.svd(x_cov)
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
            for j in range(params["max_length"]):
                k = (j + self.ofs) % (end_ix - cur_ix)
                musicVAE.y_train[i, j] = musicVAE.data.y_samples[cur_ix + k]
            cur_ix = end_ix
        assert(end_ix == musicVAE.data.num_samples)
        self.ofs += 1

    def on_epoch_end(self, epoch, logs={}):
        if params["write_history"]:
            plotScores(train_loss, '../History/BattleTheme/Scores.png', True)
        else:
            plotScores(train_loss, '../model/Scores.png', True)

        if epoch in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450] or (epoch % 100 == 0):
            write_dir = ''
            if params["write_history"]:
                write_dir = '../History/BattleTheme/e' + str(epoch)
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)
                write_dir += '/'
                model.save('../History/BattleTheme/model.h5')
                encoder.save('../History/BattleTheme/encoder_model.h5')
                pre_encoder.save('../History/BattleTheme/pre_encoder_model.h5')

                model.summary()

            else:
                model.save('../model/model.h5')
            print("Saved")

            y_song = model.predict(y_test_song, batch_size=params["batch_size"])[0]
            util.samples_to_pics(write_dir + 'test', y_song)
            midi.samples_to_midi(y_song, write_dir + 'test.mid', 16)

            make_rand_songs_normalized(write_dir, rand_vecs)

callback = CustomCallback()

print("train")
model.fit(
    x=musicVAE.y_train,
    y=musicVAE.y_train,
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    callbacks=[callback],
)

for i in range(10):
    print("test")
    util.samples_to_pics('samples/' + 'test' + str(i) , musicVAE.y_train[i])
    midi.samples_to_midi(musicVAE.y_train[i], 'samples/testtest' + str(i) +'.mid', 96)
