import os
import cv2
import numpy as np
import matplotlib as plt
import imutils
import random
import sys
from matplotlib import pyplot as plt


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

def Img_Contrast(gray):

    # hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    #
    # cdf = hist.cumsum()
    #
    # # cdf의 값이 0인 경우는 mask처리를 하여 계산에서 제외
    # # mask처리가 되면 Numpy 계산에서 제외가 됨
    # # 아래는 cdf array에서 값이 0인 부분을 mask처리함
    # cdf_m = np.ma.masked_equal(cdf, 0)
    #
    # # History Equalization 공식
    # cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    #
    # # Mask처리를 했던 부분을 다시 0으로 변환
    # cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    #
    # img2 = cdf[img]
    # plt.subplot(121), plt.imshow(img), plt.title('Original')
    # plt.subplot(122), plt.imshow(img2), plt.title('Equalization')
    # plt.show()
    # print(img.shape)
    # print(img2.shape)





    # OpenCV의 Equaliztion함수
    height, width = gray.shape[:2]
    print("test height = {0} width = {1}".format(height,width))

    result = cv2.equalizeHist(gray.copy())
    gray = cv2.resize(gray, (width * 2, height * 2))
    result = cv2.resize(result, (width * 2, height * 2))
    cv2.imshow("before_result", result)


    result = cv2.blur(result[int(height * 0.4) : int(height * 0.6), int(width * 0.3) : int(width * 0.7)]
                      ,(9, 9), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

    cv2.imshow("contrast_gray",gray)
    cv2.imshow("contrast_result",result)


    return result





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

    orig = img.copy()                            # 복사
    cv2.imshow("origin",img)
    # img = Img_Contrast(img)
    r= 800.0/img.shape[0]                        # 이건 먼지 모름..
    dim = (int(img.shape[1] * r), 800)           # 이건 먼지 모름..

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환 캐니엣지는 이진화된 이미지만 입력받을 수 있음 즉 채널2개
    gray = Img_Contrast(gray)
    gray = cv2.GaussianBlur(gray,(3,3),0)        # 노이즈 제거를 위한 가우시안블러

    cv2.imshow("grayscale",gray)
    cv2.imwrite("1111.jpg",gray)
    edged = cv2.Canny(gray,30,200)              # 엣지검출 이미지, 최소경계값, 최대경계값

    cv2.imshow("cannyedge",edged)

    # (cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    #
    # for c in cnts:
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c,0.04 * peri, True)
    #
    #     if len(approx) == 4:
    #         screenCnt = approx
    #         break
    #
    # cv2.drawContours(img,[screenCnt], -1 , (0,255,0),2)
    # cv2.imshow("outline", img)
    #
    # rect = Order_Point(screenCnt.reshape(4,2))  # delete r !!!!
    # (topleft, topright, bottomright, bottomleft) = rect
    #
    # w1 = abs(bottomright[0] - bottomleft[0])
    # w2 = abs(topright[0] - topleft[0])
    # h1 = abs(topright[1] - bottomright[1])
    # h2 = abs(topleft[1] - bottomleft[1])
    #
    # # result = img[int(topright[1] * r): int(bottomright[1] * r), int(topleft[0] * r): int(topright[0] * r)] #height, width
    # result = orig[int(topright[1]): int(bottomright[1]), int(topleft[0]): int(topright[0])] #height, width
    # cv2.imshow("resultplz",result)
    #
    # maxwidth = max([w1,w2])
    # maxheight = max([h1,h2])
    # dst = np.float32([[0,0], [maxwidth -1 ,0], [maxwidth - 1, maxheight - 1],[0,maxheight - 1]])
    # M = cv2.getPerspectiveTransform(rect, dst)
    # warped = cv2.warpPerspective(orig, M, (maxwidth, maxheight))
    #
    # cv2.imshow("warped", warped)

def Print_Img(sliced_list):

    for i in range(len(sliced_list)):
        name = "sliced_img_num" + str(i + 1)
        cv2.imshow(name,sliced_list[i])

if __name__ == "__main__":
                                                                                 # 한글로된 부분만 보세요
    img = cv2.imread("/home/dokyun/carplate_testimg/plz.jpg", cv2.IMREAD_COLOR)  # 이미지 열어 온다
    print(type(img))
    print(img.shape)
    height, width = img.shape[:2]
    print(height, width)  # height = 177, width = 640
    temp = img.copy()
    #slice_list = IMG_Slice(temp)
    #Detecting_Edge(temp)                                                         # 엣지검출 함수 진입





    temp = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
    Img_Contrast(temp)





    cv2.waitKey(0)
    cv2.destroyAllWindows()
