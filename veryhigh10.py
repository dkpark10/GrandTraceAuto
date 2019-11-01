import cv2
import os,sys

if __name__ == '__main__':

    path = '/home/bit/veryhigh.png'
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    height, width = img.shape[:2]
    gap = int(width / 10)

    for idx in range(10):

        sliced = img[int(0):int(height), int(gap * idx): int(gap * (idx + 1))]
        sliced = cv2.resize(sliced, dsize = (64,64))
        name = '/home/bit/dkNAG/veryhigh_'

        if idx == 9:
            cv2.imwrite(name + str(0) + '.png', sliced,[cv2.IMWRITE_PNG_COMPRESSION, 0])
        else :
            cv2.imwrite(name + str(idx + 1) + '.png', sliced,[cv2.IMWRITE_PNG_COMPRESSION, 0])




