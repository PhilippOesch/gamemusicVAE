from keras.models import Model, Sequential, load_model
from keras import backend as K

VAE_B1 = 0.02
VAE_B2 = 0.1
EPOCHS = 2000
BATCH_SIZE = 200
WRITE_HISTORY = True
NUM_RAND_SONGS = 50
PLAY_ONLY = True

params = {
    "use_embedding": False,
    "param_size": 120,
    "activation_str": 'relu',
    "max_length": 16,
    "bn_m": 0.9,
    "do_rate": 0.1,
    "lr": 0.001
}

model= load_model('model.h5', compile = False)

func = K.function([model.get_layer('encoder').input, K.learning_phase()],
                  [model.layers[-1].output])
enc = Model(inputs=model.input, outputs=model.get_layer('pre_encoder').output)

rand_vecs = np.random.normal(0.0, 1.0, (NUM_RAND_SONGS, params["param_size"]))
np.save('rand.npy', rand_vecs)

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

model = load_model('model.h5', compile=False)

if not os.path.exists('TestSamples'):
    os.makedirs('TestSamples')

write_dir = 'TestSamples/'
print("Generating Songs...")
make_rand_songs_normalized('', rand_vecs)
for i in xrange(20):
    x_test_song = x_train[i:i+1]
    y_song = model.predict(x_test_song, batch_size=BATCH_SIZE)[0]
    midi.samples_to_midi(y_song, 'TestSamples'+'gt' + str(i) + '.mid', 16)
exit(0)
