import numpy as np
import cv2
import sys
import os
from random import randint


def Cv2_Waitkey():

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindow()
    elif cv2.waitKey(0) == ord('s'):
        cv2.destroyAllWindow()


def Check_Ndarray_Label(ndarray, label):

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
            temp = cv2.hconcat([temp,img])
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

    result = cv2.resize(result, dsize=(640,640))
    cv2.imshow("check",result)
    Label_Print(check_label)
    Cv2_Waitkey()


def Label_Print(check_label):

    for i in range(len(check_label)):
        for j in range(len(check_label[i])):
            print(check_label[i][j], end = " ")
        print()

if __name__ == '__main__':

    print("main exec")

    aa = 'hr_ndarray_ver1.npy'
    bb = 'hr_label_ver1.npy'
    ndarray = np.load(sys.argv[1])
    label = np.load(sys.argv[2])
    print(label.shape)
    print(ndarray.shape)

    Check_Ndarray_Label(ndarray, label)
