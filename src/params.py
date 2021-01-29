from tensorflow.keras.optimizers import Adam, SGD

VAEparams = {
    "optimizer": Adam,
    "use_embedding": False,
    "param_size": 120,
    "activation_str": 'relu',
    "max_length": 16,
    "bn_m": 0.9,
    "do_rate": 0.1,
    "lr": 0.00075,
    "vae_b1": 0.02,
    "vae_b2": 0.1,
    "epochs": 500,
    "batch_size": 350,
    "write_history": False,
    "num_rand_songs": 100,
    "play_only": False,
    "history_dir": '../History/overworld_themes_model7_transposed/',
    "log_dir": '../tensorboard',
    "continue_training": False,
    "createTestingValues": True
}