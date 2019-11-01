import os
import sys
import cv2
from math import ceil

def cut_10_left(img):

    height, width = img.shape[:2]
    ret = img[int(0):int(height), int(ceil(width * 0.1)):int(width)]

    return ret


def cut_10_up(img):

    height,width = img.shape[:2]
    if height >= 9:
        ret = img[int(ceil(0.2 * height)): int(height), int(0):int(width)]
    else:
        ret = img[int(ceil(0.1 * height)): int(height), int(0):int(width)]

    return ret


def cut_10_right(img):

    height,width = img.shape[:2]
    ret = img[int(0):int(height),int(0):int(ceil(width * 0.9))]

    return ret

if __name__ == '__main__':

    path = "/home/bit/final_cp_dataset/ver5temp/"

    foldername = sys.argv[1]
    #savepath = "/home/bit/final_cp_dataset/ver3pzz(finaldktemp)/LR/dataset_9/"
    savenum = sys.argv[2]
    howcut = sys.argv[3] 

    savenum = int(savenum)

    path += foldername

    pathlist = os.listdir(path)
    for idx in range(len(pathlist)):
    
        imgname = path + "/" + pathlist[idx]
        img = cv2.imread(imgname, cv2.IMREAD_COLOR)
        #savename = savepath + savepath[-2] + "_" + str(savenum) + ".png"
        
        if howcut == '0':
            #cv2.imwrite(savename,img,[cv2.IMWRITE_PNG_COMPRESSION,0])
            cv2.imwrite(imgname,img,[cv2.IMWRITE_PNG_COMPRESSION,0])
            savenum += 1 

        elif howcut == 'left':
            croped_img = cut_10_left(img)    
            #cv2.imwrite(savename,croped_img,[cv2.IMWRITE_PNG_COMPRESSION,0])
            cv2.imwrite(imgname,croped_img,[cv2.IMWRITE_PNG_COMPRESSION,0])
            savenum += 1
        
        elif howcut == 'up':
            croped_img = cut_10_up(img)    
            #cv2.imwrite(savename,croped_img,[cv2.IMWRITE_PNG_COMPRESSION,0])
            cv2.imwrite(imgname,croped_img,[cv2.IMWRITE_PNG_COMPRESSION,0])
            savenum += 1 

        elif howcut == 'right':
            croped_img = cut_10_right(img)
            #cv2.imwrite(savename,croped_img,[cv2.IMWRITE_PNG_COMPRESSION,0])
            cv2.imwrite(imgname,croped_img,[cv2.IMWRITE_PNG_COMPRESSION,0])
            savenum += 1 

