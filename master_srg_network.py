from keras.applications import VGG19
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
from master_srg_network import Generator_Model, Discriminator_Model, SRGAN, VGG19_Model
import datetime
from data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os


class VGG19_Model(object):
    def __init__(self, width, height, channels):

        self.hr_shape = (height, width, channels)
        self.vgg = VGG19(weights='imagenet')
        # set outputs to outputs of last conv layer in block3
        self.vgg.outputs = [self.vgg.layers[9].output]

        self.img = Input(shape=self.hr_shape)

        # Extract image features
        self.img_features = self.vgg(self.img)

        self.vgg_model = Model(self.img, self.img_features)

class Generator_Model(object):
    def __init__(self, lr_width=64, lr_height=64, channels=3, filters=64, 
                residual_blocks=16):

        # Low resolution image input
        self.lr_shape = (lr_height, lr_width, channels)
        self.img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        self.c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(self.img_lr)
        self.c1 = Activation('relu')(self.c1)

        # Propogate through residual blocks
        self.r = self.residual_block(self.c1, filters)
        for _ in range(residual_blocks-1):
            self.r = self.residual_block(self.r, filters)

        # Post-residual block
        self.c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(self.r)
        self.c2 = BatchNormalization(momentum=0.8)(self.c2)
        self.c2 = Add()([self.c2, self.c1])

        # Upsampling
        self.u1 = self.deconv2d(self.c2)
        self.u2 = self.deconv2d(self.u1)

        # Generate high resolution output (SR)
        self.gen_hr = Conv2D(channels, kernel_size=9, strides=1, padding='same',
                activation='tanh')(self.u2)

        self.generator_model =  Model(self.img_lr, self.gen_hr)


    def residual_block(self, layer_input, filters):

        # Residual block described in paper
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = Activation('relu')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, layer_input])
        return d

    def deconv2d(self, layer_input):

        # Layers used during upsampling
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('relu')(u)
        return u

class Discriminator_Model(object):
    def __init__(self, width, height, channels, filters):

        self.hr_shape = (height, width, channels)

        # Input img
        self.d0 = Input(shape=self.hr_shape)

        self.d1 = self.d_block(self.d0, filters, bn=False)
        self.d2 = self.d_block(self.d1, filters, strides=2)
        self.d3 = self.d_block(self.d2, filters*2)
        self.d4 = self.d_block(self.d3, filters*2, strides=2)
        self.d5 = self.d_block(self.d4, filters*4)
        self.d6 = self.d_block(self.d5, filters*4, strides=2)
        self.d7 = self.d_block(self.d6, filters*8)
        self.d8 = self.d_block(self.d7, filters*8, strides=2)

        #self.fl0 = Flatten()(self.d8)
        #self.d9 = Dense(filters*16)(self.fl0)
        self.d9 = Dense(filters*16)(self.d8)
        self.d10 = LeakyReLU(alpha=0.8)(self.d9)
        self.validity = Dense(1, activation='sigmoid')(self.d10)

        self.discriminator_model = Model(self.d0, self.validity)

    def d_block(self, layer_input, filters, strides=1, bn=True):

        # Discriminator layer
        d = Conv2D(filters, kernel_size=3, strides=strides,
                padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

class SRGAN(object):
    def __init__(self, discriminator_model, generator_model, vgg_model,
            hr_width, hr_height, lr_width, lr_height, channels):

        self.hr_shape = (hr_height, hr_width, channels)
        self.lr_shape = (lr_height, lr_width, channels)

        self.optimizer = Adam(0.0002, 0.5)

        self.D_model = discriminator_model
        self.G_model = generator_model
        self.vgg_model = vgg_model

        #self.vgg_model.compile(loss='mse', optimizer=optimizer,
                               #metrics=['accuracy'])
        #self.D_model.compile(loss='mse', optimizer=optimizer,
                               #metrics=['accuracy'])
        # The Generator model will be compiled later  within SRGAN architecture
        
        self.img_hr = Input(shape=self.hr_shape)
        self.img_lr = Input(shape=self.lr_shape)
        self.fake_hr = self.G_model(self.img_lr)
        self.fake_features = self.vgg_model(self.fake_hr)

        self.D_model.trainable = False

        self.validity = self.D_model(self.fake_hr)

        self.srgan_model = Model([self.img_lr],
                                 [self.validity, self.fake_features])
        #self.srgan_model.compile(loss=['binary_crossentropy', 'mse'],
                                 #loss_weights=[1e-3, 1],
                                 #optimizer=optimizer)

