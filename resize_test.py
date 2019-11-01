from keras.datasets import mnist
import numpy as np
import cv2

(xt, yt), (xv, yv) = mnist.load_data()

for i in range(len(xt)):
    resized = cv2.resize(xt[i], dsize=(64,64))
    #cv2.imwrite('rm'+str(i)+'.png', resized)
    print(i)

resized = cv2.resize(xt[3], dsize=(64,64))
print(resized)
print(xt[3])

print(np.mean(resized))
print(np.mean(xt[3]))

print(np.mean(xt))

