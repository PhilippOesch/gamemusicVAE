from params import VAEparams
import network.musicVAE as musicVAE
import network.musicVAEtest as musicVAEtest
import numpy as np

rand_vecs = np.random.normal(
    0.0, 1.0, (VAEparams["num_rand_songs"], VAEparams["param_size"]))

encoder1= musicVAE.get_encoder(False, 120, 'relu')
encoder2= musicVAEtest.get_pre_encoder(False, 120, 'relu', 16)

x_enc1= np.squeeze(encoder1.predict(musicVAE.data.y_orig))[1]
x_enc2= np.squeeze(encoder2.predict(musicVAE.data.y_orig))

print("encoder1_shape: ", x_enc1.shape)
print("encoder2_shape: ", x_enc2.shape)
