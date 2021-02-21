from params import GeneralParams, VAEparams
import numpy as np
import utility.midi as midi

# rand_vecs = np.random.normal(
#     0.0, 1.0, (GeneralParams["num_rand_songs"], VAEparams["param_size"]))

# # encoder1= musicVAE.get_encoder(False, 120, 'relu')
# # encoder2= musicVAEtest.get_pre_encoder(False, 120, 'relu', 16)

# # x_enc1= np.squeeze(encoder1.predict(musicVAE.data.y_orig))[1]
# # x_enc2= np.squeeze(encoder2.predict(musicVAE.data.y_orig))

# # print("encoder1_shape: ", x_enc1.shape)
# # print("encoder2_shape: ", x_enc2.shape)

# print (convMusicVAE.y_train.shape)
# print (convMusicVAE.y_train[0, 0])

model1name = "overworld_theme"
val_dir1 = "../evaluation/evaluation_sets/" + model1name
train_set_model1 = np.load(val_dir1 + '/train_set_samples.npy')
midi.samples_to_midi(train_set_model1[0], '../TestSamples/ttttttest.mid', 16, 0.25)

