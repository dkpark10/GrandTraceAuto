import numpy as np
import os
import sys
import cv2

if __name__ == '__main__':


    for idx in range(10):
        
        validlist = []
        testlist = []

        path = '/home/bit/final_cp_dataset/ver5/'
        title = str(idx) + "_lr_ndarray.npy"
        path += title
       
        ndarr = np.load(path)
        np.random.shuffle(ndarr)

        begin = 0
        end = len(ndarr)
        mid = int(0.2 * end)

        for jdx in range(begin, mid):
        
            validlist.append(ndarr[jdx])

        validnp = np.array(validlist)
        validtitle = '/home/bit/final_cp_dataset/ver5v/' + str(idx) + '_valid_lr_ndarray.npy'
        np.save(validtitle, validnp)
            
        for kdx in range(mid, end):
            
            testlist.append(ndarr[kdx])

        testnp = np.array(testlist)
        testtitle = '/home/bit/final_cp_dataset/ver5t/' + str(idx) + '_test_lr_ndarray.npy'
        np.save(testtitle, testnp)



        

        
