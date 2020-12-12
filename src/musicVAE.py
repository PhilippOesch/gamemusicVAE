import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Lambda, TimeDistributed, Reshape, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.backend import learning_phase
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import *
from data import *
from tensorflow.keras.backend import learning_phase


class MusicVAE():
    def __init__(self, optimizer, use_embedding, param_size, activation_str, max_length, bn_m, do_rate, lr, vae_b1, vae_b2, epochs, batch_size, write_history, num_rand_songs, play_only):
        self.optimizer= optimizer
        self.use_embedding= use_embedding
        self.param_size= param_size
        self.activation= activation_str
        self.max_length= max_length
        self.bn_m= bn_m
        self.do_rate= do_rate
        self.lr= lr
        self.vae_b1= vae_b1
        self.vae_b2= vae_b2
        self.epochs= epochs
        self.batch_size= batch_size
        self.write_history= write_history
        self.num_rand_songs= num_rand_songs
        self.play_only= play_only

        self.data= data = Data(use_embedding=False)
        self.x_train, self.y_train, self.x_shape, self.y_shape = self.data.get_train_set()

    def get_pre_encoder(self):
        if self.use_embedding:
            x_in = Input(shape= self.x_shape[1:])
            x = Embedding(self.x_train.shape[0], self.param_size, input_length=1)(x_in)
            x= Flatten(name='pre_encoder')(x)
        else:
            x_in= Input(shape= self.y_shape[1:])
            x= Reshape((self.y_shape[1], -1))(x_in)
            
            x = TimeDistributed(Dense(2000, activation= self.activation))(x)

            x = TimeDistributed(Dense(200, activation= self.activation))(x)

            x= Flatten()(x)

            x= Dense(1600, activation= self.activation)(x)

            self.z_mean= Dense(self.param_size)(x)
            self.z_log_sigma_sq= Dense(self.param_size)(x)
            x_out= Lambda(self.vae_sampling, output_shape=(self.param_size,), name= 'pre_encoder')([self.z_mean, self.z_log_sigma_sq])

            model= Model(x_in, x_out)

            plot_model(model, to_file='../model/pre_encoder_model.png', show_shapes=True)

            return model
            

    def get_encoder(self, learning_phase= False):
        x_in = Input(name= 'encoder', shape= (self.param_size,))
        x = Dense(1600)(x_in)
        x = BatchNormalization(momentum= self.bn_m)(x)
        x = Activation(self.activation)(x)
        if self.do_rate> 0:
            x= Dropout(self.do_rate)(x)

        x = Dense(self.max_length * 200)(x)
        x = Reshape((self.max_length, 200))(x)
        x = TimeDistributed(BatchNormalization(momentum= self.bn_m))(x)
        x = Activation(self.activation)(x)
        if self.do_rate> 0:
            x = Dropout(self.do_rate)(x)

        x = TimeDistributed(Dense(2000))(x)
        x = TimeDistributed(BatchNormalization(momentum= self.bn_m))(x)
        x = Activation(self.activation)(x)
        if self.do_rate> 0:
            x= Dropout(self.do_rate)(x)
        
        x= TimeDistributed(Dense(self.y_shape[2]* self.y_shape[3], activation= 'sigmoid'))(x)
        x= Reshape((self.y_shape[1], self.y_shape[2], self.y_shape[3]))(x)

        if learning_phase:
            model= Model(x_in, x)
            # model= K.function([x_in], [x])
        else:
            model= Model(x_in, x)
            plot_model(model, to_file='../model/encoder_model.png', show_shapes=True)

        return model


    def build_full_model(self):
        if self.play_only:
            self.encoder= load_model('../model/encoder_model.h5')
            self.pre_encoder= load_model('../model/pre_encoder_model.h5')
        else:
            self.pre_encoder= self.get_pre_encoder()
            self.encoder= self.get_encoder()

        x_in = self.pre_encoder.input
        x_out = self.encoder(self.pre_encoder.output)
        self.model= Model(x_in, x_out)
        self.model.compile(optimizer= self.optimizer(lr= self.lr), loss= self.vae_loss, experimental_run_tf_function=False)

        plot_model(self.model, to_file='../model/model.png', show_shapes=True)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss= binary_crossentropy(x, x_decoded_mean)
        kl_loss= self.vae_b2* K.mean(1+ self.z_log_sigma_sq- K.square(self.z_mean)- K.exp(self.z_log_sigma_sq), axis= None)
        return xent_loss - kl_loss

    def vae_sampling(self, args):
        self.z_mean, self.z_log_sigma_sq = args
        epsilon = K.random_normal(shape=K.shape(self.z_mean), mean=0.0, stddev=self.vae_b1)
        return self.z_mean + K.exp(self.z_log_sigma_sq * 0.5) * epsilon


    
