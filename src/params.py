from tensorflow.keras.optimizers import Adam, SGD

VAEparams = {
    "optimizer": Adam,
    "param_size": 200,
    "activation_str": 'relu',
    "max_length": 16,
    "bn_m": 0.9,
    "do_rate": 0.1,
    "lr": 0.00075,
    "epochs": 2000,
    "batch_size": 200,
    "use_batchnorm": True,
    "ignore_encoder_layer2": False,
}

GeneralParams= {
    "vae_b1": 1,
    "vae_b2": 5,
    "model_name": "model12",
    "write_history": True,
    "num_rand_songs": 10,
    "use_batchnorm": False,
    "play_only": False,
    "history_dir": '../History/battletheme_more_data_refacotored_model_beta5/',
    "log_dir": '../tensorboard',
    "continue_training": False,
    "createTestingValues": False,
}
