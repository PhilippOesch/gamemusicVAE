from tensorflow.keras.optimizers import Adam, SGD

VAEparams = {
    "optimizer": Adam,
    "param_size": 200,
    "activation_str": 'relu',
    "max_length": 16,
    "bn_m": 0.9,
    "do_rate": 0.1,
    "lr": 0.00075,
    "epochs": 50,
    "batch_size": 200,
    "use_batchnorm": False,
    "ignore_encoder_layer2": False
}

GeneralParams= {
    "model_name": "model12",
    "write_history": True,
    "num_rand_songs": 10,
    "use_batchnorm": False,
    "play_only": False,
    "history_dir": '../History/battletheme_more_data_refacotored_model/',
    "log_dir": '../tensorboard',
    "continue_training": False,
    "createTestingValues": False,
}
