#!/usr/bin/env python3

import sys
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU


class Trainer:
    def __init__(self, width=28, height=28, channels=1, latent_size=100, epochs=50000, batch=32, checkpoint=50,
                 model_type=-1):

        self.W = width
        self.H = height
        self.C = channels
        self.EPOCHS = epochs
        self.BATCH = batch
        self.CHECKPOINT = checkpoint
        self.model_type = model_type

        self.LATENT_SPACE_SIZE = latent_size

        self.generator = Generator(height=self.H, width=self.W, channels=self.C, latent_size=self.LATENT_SPACE_SIZE)
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)

        self.load_MNIST()

    def load_MNIST(self, model_type=3):

        allowed_types = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if self.model_type not in allowed_types:
            print('ERROR: Only Integer Values from -1 to 9 are allowed')

        (self.X_train, self.Y_train), (_, _) = mnist.load_data()
        if self.model_type != -1:
            self.X_train = self.X_train[np.where(self.Y_train == int(self.model_type))[0]]

        # Rescale -1 to 1
        # Find Normalize Function from CV Class
        self.X_train = (np.float32(self.X_train) - 127.5) / 127.5
        self.X_train = np.expand_dims(self.X_train, axis=3)
        return

    def train(self):

        for e in range(self.EPOCHS):
            # Train Discriminator
            # Make the training batch for this model be half real, half noise
            # Grab Real Images for this training batch
            count_real_images = int(self.BATCH / 2)                                 # 16
            starting_index = randint(0, (len(self.X_train) - count_real_images))    # 0, 60000 - 16
            
            real_images_raw = self.X_train[starting_index: (starting_index + count_real_images)] 
            #starting_index, starting_index + 16
            
            x_real_images = real_images_raw.reshape(count_real_images, self.W, self.H, self.C) # 16,64,64,3    
            y_real_labels = np.ones([count_real_images, 1])                                    # np.ones (16,1)

            # Grab Generated Images for this training batch
            latent_space_samples = self.sample_latent_space(count_real_images)
            x_generated_images = self.generator.Generator.predict(latent_space_samples)        #  모델 사용
            y_generated_labels = np.zeros([self.BATCH - count_real_images, 1])                 # np.zeros(16,1)

            # Combine to train on the discriminator
            x_batch = np.concatenate([x_real_images, x_generated_images])
            y_batch = np.concatenate([y_real_labels, y_generated_labels])

            # Now, train the discriminator with this batch
            discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch, y_batch)[0]

            # Generate Noise
            x_latent_space_samples = self.sample_latent_space(self.BATCH)
            y_generated_labels = np.ones([self.BATCH, 1])
            generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples, y_generated_labels)

            print('Epoch: ' + str(int(e)) + ', [Discriminator :: Loss: ' + str(
                discriminator_loss) + '], [ Generator :: Loss: ' + str(generator_loss) + ']')

            if e % self.CHECKPOINT == 0:
                self.plot_checkpoint(e)
        return

    def sample_latent_space(self, instances):
        return np.random.normal(0, 1, (instances, self.LATENT_SPACE_SIZE))              # 0,1,(16 ~ 100)

    def plot_checkpoint(self, e):
        filename = "data/sample_" + str(e) + ".png"

        noise = self.sample_latent_space(16)
        images = self.generator.Generator.predict(noise)

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.H, self.W])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return


class GAN(object):

    def __init__(self, discriminator, generator):
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

        self.Generator = generator

        self.Discriminator = discriminator
        self.Discriminator.trainable = False

        self.gan_model = self.model()
        self.gan_model.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        self.save_model()
        self.summary()

    def model(self):
        model = Sequential()
        model.add(self.Generator)
        model.add(self.Discriminator)
        return model

    def summary(self):
        return self.gan_model.summary()

    def save_model(self):
        plot_model(self.gan_model.model, to_file='data/GAN_Model.png')


class Generator(object):

    def __init__(self, width=28, height=28, channels=1, latent_size=100):
        self.W = width
        self.H = height
        self.C = channels
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

        self.LATENT_SPACE_SIZE = latent_size                                    # 100
        self.latent_space = np.random.normal(0, 1, (self.LATENT_SPACE_SIZE,))

        self.Generator = self.model()
        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        self.save_model()
        self.summary()

    def model(self, block_starting_size=128, num_blocks=4):
        model = Sequential()

        block_size = block_starting_size
        model.add(Dense(block_size, input_shape=(self.LATENT_SPACE_SIZE,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        for i in range(num_blocks - 1):
            block_size = block_size * 2
            model.add(Dense(block_size))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(self.W * self.H * self.C, activation='tanh'))
        model.add(Reshape((self.W, self.H, self.C)))

        return model

    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        plot_model(self.Generator.model, to_file='data/Generator_Model.png')


class Discriminator(object):
    def __init__(self, width = 28, height= 28, channels = 1, latent_size=100):
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)


        self.Discriminator = self.model()
        self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'] )
        self.save_model()
        self.summary()

    def model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.SHAPE))
        model.add(Dense(self.CAPACITY, input_shape=self.SHAPE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int(self.CAPACITY/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def summary(self):
        return self.Discriminator.summary()

    def save_model(self):
        plot_model(self.Discriminator.model, to_file='data/Discriminator_Model.png')


if __name__ == '__main__':

    HEIGHT = 28
    WIDTH = 28
    CHANNEL = 1
    LATENT_SPACE_SIZE = 100
    EPOCHS = 50001
    BATCH = 32
    CHECKPOINT = 500
    MODEL_TYPE = -1

    trainer = Trainer(height=HEIGHT, \
                      width=WIDTH, \
                      channels=CHANNEL, \
                      latent_size=LATENT_SPACE_SIZE, \
                      epochs=EPOCHS, \
                      batch=BATCH, \
                      checkpoint=CHECKPOINT,
                      model_type=MODEL_TYPE)

    trainer.train()
