import numpy as np
import cv2
import sys
import os
from random import randint
from keras.applications import VGG19
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
# from master_srg_network import Generator_Model, Discriminator_Model, SRGAN, VGG19_Model
import datetime
from data_loader import DataLoader
import matplotlib.pyplot as plt

class Checker(object):

    def __init__(self, ndarray, label):
        self.ndarray = ndarray
        self.label = label
        self.Check_Ndarray_Label(self.ndarray, self.label)

    def Check_Ndarray_Label(self, ndarray, label):
        columnlist = []
        check_label = []

        for i in range(10):

            init = randint(0, len(label))
            temp = ndarray[init]
            temp_label = []

            if sys.argv[1][:2] == 'hr':
                temp_label.append(np.argmax(label[init]))
            else:
                temp_label.append(np.max(label[init]))

            for j in range(9):

                rannum = randint(0, len(label))
                img = ndarray[rannum]
                temp = cv2.hconcat([temp, img])
                if sys.argv[1][:2] == 'hr':
                    temp_label.append(np.argmax(label[rannum]))
                else:
                    temp_label.append(np.max(label[rannum]))

            columnlist.append(temp)
            check_label.append(temp_label)

        result = columnlist[0]
        for i in range(9):
            temp = columnlist[i + 1]
            result = cv2.vconcat([result, temp])

        result = cv2.resize(result, dsize=(640, 640))
        cv2.imshow("check", result)
        self.Label_Print(check_label)
        self.Cv2_Waitkey()


    def Label_Print(self, check_label):
        for i in range(len(check_label)):
            for j in range(len(check_label[i])):
                print(check_label[i][j], end=" ")
            print()


    def Cv2_Waitkey(self):
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindow()
        elif cv2.waitKey(0) == ord('s'):
            cv2.destroyAllWindow()




if __name__ == '__main__':
    print("main exec")
    datapath = '/home/bit/final_cp_dataset/ver4/'
    ndarray = np.load(datapath + sys.argv[1])
    label = np.load(datapath + sys.argv[2])

    Checker = Checker(ndarray, label)
