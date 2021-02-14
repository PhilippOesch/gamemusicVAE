from tensorflow.keras.optimizers import Adam, SGD

VAEparams = {
    "optimizer": Adam,
    "use_embedding": False,
    "param_size": 120,
    "activation_str": 'relu',
    "max_length": 16,
    "bn_m": 0.9,
    "do_rate": 0.1,
    "lr": 0.0008,
    "vae_b1": 0.02,
    "vae_b2": 0.1,
    "epochs": 2000,
    "batch_size": 250,
    "write_history": True,
    "num_rand_songs": 10,
    "play_only": False,
    "history_dir": '../History/battletheme_more_data_refacotored_model/',
    "log_dir": '../tensorboard',
    "continue_training": False,
    "createTestingValues": False
}