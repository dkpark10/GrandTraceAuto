import numpy as np
import cv2

def batch_generator(X, Y, batch_size):
    
    indices = np.arange(len(X))
    rest = indices
    batch = []
    while True:
        
        np.random.shuffle(indices)

        for i in indices:
            batch.append(i)
            if rest < batch_size:
                print("test")

            if len(batch) == batch_size:
                yield (X[batch], Y[batch])
                rest -= batch_size 
                batch = []


def batch_gene(x,y,size):

    indices = np.arange(len(x))
    rest = len(indices)
    batch = []
    np.random.shuffle(indices)

    for i in indices:
        
        batch.append(i)
        if rest < size:
            temp = indices[-rest:]
            yield (x[temp] , y[temp])
            break

        if len(batch) == size:
            
            yield (x[batch], y[batch])
            rest -= size
            batch = []





a = np.asarray([111,222,333,444,555,666,777,888])
b = np.asarray([1,2,3,4,5,6,7,8])

bg = batch_generator(a,b,3)
bbgg = batch_gene(a,b,3)

i = 0
for x,y in bbgg:

    print(x,y)




    
