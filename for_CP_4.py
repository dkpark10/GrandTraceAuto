import argparse
parser = argparse.ArgumentParser(description='ArgParser')
parser.add_argument(
        '--tryout', action='store', dest='tryout',
        default=10001)
parser.add_argument(
        '--epochs', action='store', dest='epochs',
        default=1)
parser.add_argument(
        '--batch_size', action='store', dest='batch_size',
        default=16)
parser.add_argument(
        '--learning_rate', action='store', dest='learning_rate',
        default=0.0001)
parser.add_argument(
        '--checkpoint', action='store', dest='checkpoint',
        default=None)
parser.add_argument(
        '--samples', action='store', dest='samples',
        default=512)
parser.add_argument(
        '--scale', action='store', dest='scale',
        default=2)
parser.add_argument(
        '--data', action='store', dest='data',
        default=None)
parser.add_argument(
        '--GTA_train', action='store', dest='gt',
        default=False)
parser.add_argument(
        '--teacher_train', action='store', dest='tt',
        default=False)
parser.add_argument(
        '--test_image', action='store', dest='test_image',
        default=None)
parser.add_argument(
        '--test', action='store', dest='test',
        default=False)

parsed = parser.parse_args()
lr = float(parsed.learning_rate)
bs = int(parsed.batch_size)
eps = int(parsed.epochs)
tryout = int(parsed.tryout)
samples = int(parsed.samples)
scale = int(parsed.scale)
data = parsed.data
data = './hr_image.npy'
data_label = data[:-4] + '_label.npy'
if data is None:
    if parsed.test_image is None:
        raise ValueError
test_image = parsed.test_image
chk = str(parsed.checkpoint)

import numpy as np
import cv2
from keras.models import Model
from keras.applications import VGG19
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, Adadelta
from networks import Discriminator_Model, SRGAN, Perceptual_Teacher
from rdn import RDN, RDN_m # Generator Model
from gta_utils import batch_generator, t_v_split, make_fat_lrs

##### Load Data #####################################################
print()
print('          Load Data')
print()
data = np.load(data)
data_label = np.load(data_label)
#print(data.shape)
#print(data_label.shape)
data_lr_path = './CP_LR_T/lr_image_t_0.npy' #########
data_lr_path = data_lr_path[:-5]
print(data_lr_path)
data_lr = []
for i in range(10):
    data_lr.append(make_fat_lrs(np.load(data_lr_path+str(i)+'.npy')))
data_lr = np.asarray(data_lr)
for i in range(10):
    print('Fat Tensor for Number {}'.format(i), data_lr[i].shape)
lr_shape = (32,32,12)
hr_shape = (32*scale, 32*scale, 3)

##### Define Optimizer ##############################################
print()
print('          Define Optimizer')
print()
optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
                decay=lr/2, amsgrad=False)

##### RDN ###########################################################
print()
print('          Initialize RDN (Generator)')
print()
rdn_init = RDN_m(channel=12, patch_size=32, RDB_no=20, scale=scale)
rdn = rdn_init.rdn_model
#rdn.summary()

##### Perceptual Teacher ############################################
print()
print('          Initialize Perceptually Teaching Model')
print()
c_model = Perceptual_Teacher(64,64,3,imagenet=False)
c_model.teacher_model.compile(loss=['mse'], 
                              optimizer=optimizer, 
                              metrics=['accuracy'])
interm1 = Model(inputs=c_model.teacher_model.input,
               outputs=c_model.teacher_model.get_layer('block5_conv4').output)
interm1.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

##### Discriminator ##################################################
print('          Initialize Discriminator')
print()
d_init = Discriminator_Model(64,64,3,filters=64)
print()
d_model = d_init.discriminator_model
d_model.compile(loss='mse', optimizer=Adam(0.0002,0.5), metrics=['accuracy'])
#d_model.summary()

##### SRGAN #########################################################
print()
print('          Construct SRGAN Structure')
print()
srgan_init = SRGAN(d_model, rdn, interm1, 64,64,32,32,12)
srgan_model = srgan_init.srgan_model
srgan_model.compile(loss=['binary_crossentropy', 'mse'],
                    loss_weights=[1e-3, 1],
                    optimizer=optimizer)

##### Load Weights for Models #######################################
if chk != 'None':
    print()
    print('          Loading Checkpoint')
    print()
    c_model.teacher_model.load_weights('tt_vgg19_ep20.h5')

##### Scale and Prepare the Data ###################################
print()
print('          Scale and Prepare the Data')
print()
data = data.astype('float32') / 255
x_t, y_t, x_v, y_v = t_v_split(data, data_label, 0.2)
x_lr_t = []
x_lr_v = []
for i in range(10):
    data_lr[i] = (data_lr[i].astype('float32') - 127.5) / 127.5
    data_len = len(data_lr[i])
    train_len = int(data_len * 0.8) # val_len = data_len - train_len
    x_lr_t.append(data_lr[i][:train_len])
    x_lr_v.append(data_lr[i][train_len:])
x_lr_t = np.asarray(x_lr_t)
x_lr_v = np.asarray(x_lr_v)

##### GTA Training Mode #############################################
if parsed.gt:
    batch_size_D = 4
    a_batchg_hr_D = batch_generator(x_t, batch_size=batch_size_D)
a_batch_lr_D = []
for i in range(10):
    a_batch_lr_D.append(batch_generator(x_lr_t[i], batch_size=batch_size_D))
sr_label = np.zeros((batch_size_D,) + (4,4,1))
hr_label = np.ones((batch_size_D,) + (4,4,1))

##### Discriminator Training
for i in range(1): # Train with false label
    j = np.random.randint(10)
    a_batch_sr = rdn.predict_generator(a_batch_lr_D[j], steps=1)
    d_model.fit(a_batch_sr, sr_label, epochs=1)
for i in range(1): # Train with true label
    a_batch_hr_D = next(a_batchg_hr_D)
    d_model.fit(a_batch_hr_D, hr_label, epochs=1)
d_model_pred = d_model.predict(a_batch_sr)
print('Discriminator Output: ', d_model_pred.mean())

##### Generator Training
batch_size_G = 8
a_batch_hr_G = batch_generator(x_t, y_t, batch_size=batch_size_G)
a_batch_lr_G = []
for i in range(10):
    a_batch_lr_G.append(batch_generator(x_lr_t[i], batch_size=1))
for i in range(1):
    a_batch_hr_G1, a_batch_hr_G2 = next(a_batch_hr_G) # 1:Images 2:Labels
    a_batch_label_G = []
    for _ in len(a_batch_hr_G2):
        a_batch_label_G.append(np.argmax(_)) # Label
    a_batch_hr = rdn.predict_generator(a_batch_lr_G[j], steps=1)
    a_batch_sr_feature = interm1.predict(a_batch_sr)

##### Perceptual Teacher Training Mode ##############################
if parsed.tt:
    c_model = Perceptual_Teacher(64,64,3,imagenet=False)
    c_model.base_model.trainable = True
    c_model.teacher_model.trainable = True
    c_model.teacher_model.compile(loss=['mse'], optimizer=optimizer, 
                                  metrics=['accuracy'])
    c_model.teacher_model.load_weights('tt_vgg19_ep20.h5')
    #c_model.base_model.summary()
    #c_model.teacher_model.summary()
    from math import ceil
    steps_per_epoch = ceil(len(data)/32)
    v_steps = ceil(len(data)*0.1/32)
    print('STEPS PER EPOCH', steps_per_epoch)
    a_batch_t = batch_generator(x_t, y_t, batch_size=32)
    a_batch_v = batch_generator(x_v, y_v, batch_size=32)
    for ep in [1, 2]:
        c_model.teacher_model.fit_generator(a_batch_t, 
                                steps_per_epoch=steps_per_epoch,
                                validation_data=a_batch_v, 
                                validation_steps=v_steps, epochs=20)
        #c_model.teacher_model.save_weights('tt_vgg19_ep'+str((ep+1)*20)+'.h5')
    print('LOADED')

##### Test Mode #####################################################
if parsed.test:
    c_model = Perceptual_Teacher(64,64,3,imagenet=False)
    c_model.base_model.trainable = True
    c_model.teacher_model.trainable = True
    c_model.teacher_model.compile(loss=['mse'], optimizer=optimizer, 
                                  metrics=['accuracy'])
    c_model.teacher_model.load_weights('tt_vgg19_ep20.h5')
    #c_model.base_model.summary()
    #c_model.teacher_model.summary()
    from math import ceil
    steps_per_epoch = ceil(len(data)/32)
    v_steps = ceil(len(data)*0.1/32)
    print('STEPS PER EPOCH', steps_per_epoch)
    a_batch_t = batch_generator(x_t, y_t, batch_size=32)
    a_batch_v = batch_generator(x_v, y_v, batch_size=32)
    for ep in [1, 2]:
        c_model.teacher_model.fit_generator(a_batch_t, 
                                steps_per_epoch=steps_per_epoch,
                                validation_data=a_batch_v, 
                                validation_steps=v_steps, epochs=20)
        #c_model.teacher_model.save_weights('tt_vgg19_ep'+str((ep+1)*20)+'.h5')
    print('LOADED')






