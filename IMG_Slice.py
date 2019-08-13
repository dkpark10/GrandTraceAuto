import os
import cv2
import numpy as np
import matplotlib as plt
import imutils
import random
import sys

def IMG_Slice(img):

    sliced_list = []
    height, width = img.shape[:2]
    result = img[int(height * 0.17): int(height * 0.8), int(width * 0.1): int(width * 0.91)] #height, width

    pass
    height, width = result.shape[:2]

    num1 = result[int(height * 0.1227): int(height * 0.8772), int(width * 0.0846): int(width * 0.1923)]
    sliced_list.append(num1)
    num2 = result[int(height * 0.1227): int(height * 0.8772), int(width * 0.1923): int(width * 0.3)]
    sliced_list.append(num2)
    num3 = result[int(height * 0.1227): int(height * 0.8772), int(width * 0.484): int(width * 0.5923)]
    sliced_list.append(num3)
    num4 = result[int(height * 0.1227): int(height * 0.8772), int(width * 0.5923): int(width * 0.7)]
    sliced_list.append(num4)
    num5 = result[int(height * 0.1227): int(height * 0.8772), int(width * 0.7): int(width * 0.807)]
    sliced_list.append(num5)
    num6 = result[int(height * 0.1227): int(height * 0.8772), int(width * 0.807): int(width * 0.915)]
    sliced_list.append(num6)

    return sliced_list

def Order_Point(pts):

    rect = np.zeros((4,2), dtype= "float32")
    s = pts.sum(axis = 1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts,axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return  rect

def Detecting_Edge(img):

    orig = img.copy()
    r= 800.0/img.shape[0]
    dim = (int(img.shape[1] * r), 800)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    edged = cv2.Canny(gray,75,200)

    cv2.imshow("img",img)
    cv2.imshow("edg",edged)

    (cnts,_)= cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts,key = cv2.contourArea,reverse = True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c,0.04 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    cv2.drawContours(img,[screenCnt], -1 , (0,255,0),2)
    cv2.imshow("outline", img)

    rect = Order_Point(screenCnt.reshape(4,2))  # delete r !!!!
    (topleft, topright, bottomright, bottomleft) = rect

    w1 = abs(bottomright[0] - bottomleft[0])
    w2 = abs(topright[0] - topleft[0])
    h1 = abs(topright[1] - bottomright[1])
    h2 = abs(topleft[1] - bottomleft[1])

    # result = img[int(topright[1] * r): int(bottomright[1] * r), int(topleft[0] * r): int(topright[0] * r)] #height, width
    result = orig[int(topright[1]): int(bottomright[1]), int(topleft[0]): int(topright[0])] #height, width
    cv2.imshow("resultplz",result)

    maxwidth = max([w1,w2])
    maxheight = max([h1,h2])
    dst = np.float32([[0,0], [maxwidth -1 ,0], [maxwidth - 1, maxheight - 1],[0,maxheight - 1]])
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxwidth, maxheight))

    cv2.imshow("warped", warped)

def Print_Img(sliced_list):

    for i in range(len(sliced_list)):
        name = "sliced_img_num" + str(i + 1)
        cv2.imshow(name,sliced_list[i])

if __name__ == "__main__":

    img = cv2.imread("/home/dokyun/carplate_testimg/carplate11.jpg", cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    print(height, width)  # height = 177, width = 640
    temp = img.copy()
    #slice_list = IMG_Slice(temp)
    Detecting_Edge(temp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
