from keras.optimizers import Adam
from keras.layers import Input
from master_srg_network import Generator_Model, Discriminator_Model, SRGAN, VGG19_Model
import datetime
from data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    lr_width = 64
    lr_height = 64
    hr_width = lr_width*4
    hr_height = lr_height*4
    channels = 3
    
    hr_shape = (hr_height, hr_width, channels)
    lr_shape = (lr_height, lr_width, channels)

    residual_blocks = 16

    optimizer = Adam(0.0002, 0.5)

    patch_height = int(hr_height / 2**4)
    patch_width = int(hr_width / 2**4)
    disc_patch = (patch_height, patch_width, 1)

    generator_filters = 64
    discriminator_filters = 64

    # Define VGG Model
    vgg_init = VGG19_Model(hr_width, hr_height, channels)
    vgg_model = vgg_init.vgg_model
    vgg_model.trainable = False
    vgg_model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    vgg_model.summary()

    # Define Discriminator Model
    discriminator_init = Discriminator_Model(hr_width,hr_height,
            channels,discriminator_filters)
    discriminator_model = discriminator_init.discriminator_model
    discriminator_model.compile(loss='mse', 
            optimizer=optimizer, metrics=['accuracy'])
    discriminator_model.summary()
    # Load Discriminator Weights
    discriminator_model.load_weights('./discriminator_weights_2_4000.h5')

    # Define Generator Model
    generator_init = Generator_Model(lr_width, lr_height, channels, 
            generator_filters, residual_blocks)
    generator_model = generator_init.generator_model
    generator_model.summary()
    # Load Generator Weights
    generator_model.load_weights('./generator_weights_2_4000.h5')

    # Define SRGAN Architecture
    srgan_init = SRGAN(discriminator_model, generator_model, vgg_model,
            hr_width, hr_height, lr_width, lr_height, channels)
    srgan_model = srgan_init.srgan_model
    srgan_model.compile(loss=['binary_crossentropy', 'mse'],
                        loss_weights=[1e-3, 1],
                        optimizer=optimizer)
    srgan_model.summary()

    dataset_name = 'img_align_celeba'
    data_loader = DataLoader(dataset_name=dataset_name,
                                img_res=(hr_height,hr_width))
    
    #def train(self, epochs, batch_size=1, sample_interval=50):

    start_time = datetime.datetime.now()

    epochs = 10000
    batch_size = 16
    sample_interval = 500
    model_saving_interval = 500

    def sample_images(epoch):
        os.makedirs('images/%s' % dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = generator_model.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (dataset_name, epoch))
        plt.close()

        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%s/%d_lowres%d.png' % (dataset_name, epoch, i))
            plt.close()

    for epoch in range(epochs):
        imgs_hr, imgs_lr = data_loader.load_data(batch_size)
        fake_hr = generator_model.predict(imgs_lr)
        valid = np.ones((batch_size,) + disc_patch)
        fake = np.zeros((batch_size,) + disc_patch)
        d_loss_real = discriminator_model.train_on_batch(imgs_hr, valid) # Train #
        d_loss_fake = discriminator_model.train_on_batch(fake_hr, fake) # Train #
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        imgs_hr, imgs_lr = data_loader.load_data(batch_size)
        valid = np.ones((batch_size,) + disc_patch)
        image_features = vgg_model.predict(imgs_hr)

        g_loss = srgan_model.train_on_batch([imgs_lr],[valid, image_features]) # Train #

        elapsed_time = datetime.datetime.now() - start_time

        if epoch % 10 == 0:
            print("%d time: %s" % (epoch, elapsed_time))
        if epoch % sample_interval == 0:
            sample_images(epoch)
            # Save Models Weights
            if epoch % model_saving_interval == 0:
                saved_model_name1 = 'discriminator_weights_3_' + str(epoch) + '.h5'
                saved_model_name2 = 'generator_weights_3_' + str(epoch) + '.h5'
                discriminator_model.save_weights(saved_model_name1)
                generator_model.save_weights(saved_model_name2)

