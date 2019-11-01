import argparse
import numpy as np
import cv2
from keras.models import Model
from keras.applications import VGG19
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, Adadelta
from networks import Discriminator_Model, SRGAN, Perceptual_Teacher
from networks import Generator_Model
from rdn import RDN, RDN_m # Generator Model
from gta_utils import batch_generator, t_v_split, make_fat_lrs
from random import randint

if __name__ == '__main__':

    print('main exec')
    target = randint(1,100 + 1)
    cnt = 1

    while True:

        n = int(input())
        if n == target:
            print("try {0} ~~~".format(cnt))
            break
        elif n < target:
            print("small size up")
        elif n > target:
            print("big size down")
        cnt += 1





