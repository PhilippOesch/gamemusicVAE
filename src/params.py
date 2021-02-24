from tensorflow.keras.optimizers import Adam, SGD
from tensorflow import keras

VAEparams = {
    "optimizer": Adam,
    "param_size": 256,
    "activation_str": 'relu',
    "max_length": 16,
    "bn_m": 0.9,
    "do_rate": 0.1,
    "lr": 0.00075,
    "epochs": 2000,
    "batch_size": 200,
    "use_batchnorm": True,
    "dim1": 2048,
    "dim2": 256,
    "dim3": 2048,
}

# convVAEparams = {
#     "optimizer": Adam,
#     "param_size": 512,
#     "activation_str": 'relu',
#     "max_length": 16,
#     "bn_m": 0.9,
#     "do_rate": 0.1,
#     "lr": 0.0005,
#     "epochs": 2000,
#     "batch_size": 150,
#     "use_batchnorm": True,
#     "ignore_encoder_layer2": False,
#     "dim1": 128,
#     "dim2": 2048,
# }

GeneralParams= {
    "vae_b2": 10,
    "model_name": "battle_final_1",
    "write_history": True,
    "num_rand_songs": 10,
    "use_batchnorm": False,
    "play_only": False,
    "history_dir": '../History/battle_final_1/',
    "continue_training": False,
    "createTestingValues": False,
    "dataset_name": "battle",
    "num_timesteps": 96,
    "num_notes": 88,
}
