import numpy as np
import scipy.stats as stats
import os
import cv2
import matplotlib as plt
import imutils
import random
import sys
import tarfile
import six.moves.urllib as urllib
import time
from os import listdir
import SliceImg

def Img_Kld(p, q): # p is cur q is prev

    ptemp = p.copy()
    qtemp = q.copy()

    ptemp = cv2.cvtColor(ptemp, cv2.COLOR_BGR2GRAY)
    qtemp = cv2.cvtColor(qtemp, cv2.COLOR_BGR2GRAY)
    height, width = ptemp[:2]

    qtemp = cv2.resize(qtemp, dsize=(width, height), interpolation=cv2.INTER_AREA)

    kld = stats.entropy(ptemp, qtemp)
    print(kld)
    print(len(kld))
    print(type(kld))

    sum = 0.0
    for i in range(p.shape[0]):
        sum += kld[i]

    print("sum is {0}".format(sum))
    print(type(sum))

    standard = 9
    # if sum < standard:
    # else:
    #     Create_Folder()


def Create_Folder(videoname, cpnum):

    foldername = videoname + "_" + str(cpnum)
    try:
        if not (os.path.isdir(foldername)):
            os.makedirs(os.path.join(foldername))
    except OSError as e:
        if e.errno != e.EEXIST:
            print("Create Folder Failed")
            raise


def Init_Nparr():

    ret = np.zeros((120,180,3))
    return ret

if __name__ == '__main__':

    print("main")
    temp = list()
    temp.append(4)
    print(temp[0])

    ret = np.zeros((120,180,3))
    print(ret.ndim)

    cv2.imshow("test",ret)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
