import numpy as np
import scipy.stats as stats
import os
import cv2
import matplotlib as plt
import imutils
import random
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import tarfile
import six.moves.urllib as urllib
import time
from os import listdir
import math

def Quardrisect(img):

    result = img.copy()
    sliced_list = []
    height, width = result.shape[:2]

    four = int(width/4)
    temp = 0
    
    num1 = result[int(0): int(height), int(0): int(temp + four)]; temp += four
    sliced_list.append(num1)
    num2 = result[int(0): int(height), int(temp): int(temp + four)]; temp += four
    sliced_list.append(num2)
    num3 = result[int(0): int(height), int(temp): int(temp + four)]; temp += four
    sliced_list.append(num3)
    num4 = result[int(0): int(height), int(temp): int(width)]
    sliced_list.append(num4)

    return sliced_list


def Img_Slice(img, txt):

    imgcpy = img.copy()
    line = txt.readline()
    if len(line) == 0:                           # if no box
        return imgcpy
    line = line.split(" ")

    height , width = img.shape[:2]

    box_width = float(line[3]) * width
    box_height = float(line[4]) * height
    cent_x = float(line[1]) * width
    cent_y = float(line[2]) * height

    left_x = cent_x - (box_width / 2)
    left_y = cent_y - (box_height / 2)
    right_x = cent_x + (box_width / 2)
    right_y = cent_y + (box_height / 2)

    classnum = int(line[0])

    if classnum == 0:                            # only x
        ret = imgcpy[int(0): int(height), int(left_x): int(right_x)]
    elif classnum == 1:                          # x and y
        ret = imgcpy[int(left_y): int(right_y), int(left_x): int(right_x)]

    return ret


def Create_Folder():

    foldername = "/home/bit/final_cp_dataset_zerotinine"
    numberingfold = foldername + "/" + "dataset"
    if not os.path.isdir(foldername):
        os.mkdir(foldername)

    for i in range(10):
        fold = numberingfold + "_" + str(i)
        if not os.path.idsir(fold):
            os.mkdir(fold)
        
    return foldername


if __name__ == '__main__':

    print("main exec")

    basepath = "/home/bit/Yolo_mark/x64/Release/data/testfolder"            # <-- edit folder
    filelist = os.listdir(basepath)
    txtlist = []
    savenum = [1 for i in range(10)]
    
    foldername = Create_Folder()

    for file in filelist:

        if file[-4:] == '.txt':
            txtlist.append(file)

    for idx,file in enumerate(txtlist):

        numbering = file[:4]
        temp = file[:-4]
        temp = basepath + "/" + temp + ".png"
        img = cv2.imread(temp, cv2.IMREAD_COLOR)

        txt = open(basepath + "/" + file, mode='rt', encoding='utf-8')
        ret = Img_Slice(img, txt)
        quard_list = Quardrisect(ret)

        for j in range(len(quard_list)):
            
            savetitle = foldername + "/" + "dataset_" + numbering[j] + "/" \
            numbering[j] + "_" + str(savenum[int(numbering[j])]) + ".png"    
            
            savenum[int(numbering[j])] += 1
            cv2.imwrite(savetitle, quard_list[cpnum], [cv2.IMWRITE_PNG_COMPRESSION, 0])

