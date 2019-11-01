import numpy as np

def batch_generator(X, Y=None, label=None, OHlabel=False,  batch_size=32):

    indices = np.arange(len(X))                             # 0 ~ len(x) numpy

    if Y is not None:
        if len(X) != len(Y):
            raise ValueError

    batch = []

    if label is None:
        while True:
            np.random.shuffle(indices)
            for i in indices:
                batch.append(i)
                if len(batch) == batch_size:
                    if Y is not None:
                        yield (X[batch], Y[batch])
                    else:
                        yield X[batch]
                    batch = []
    else:
        if Y is None:
            raise ValueError
        elif len(X) != len(Y):
            raise ValueError
    
        while True:
            np.random.shuffle(indices)
            i = np.random.randint(len(X))
            if np.argmax(Y[i]) == label:                    # 큰값 인덱스 리턴
                batch.append(i)
            if len(batch) == batch_size:
                if OHlabel:
                    yield (X[batch], Y[batch])
                else:
                    yield X[batch]
                batch = []



def t_v_split(X, Y, ratio=0.1):

    lenx = len(X)
    if lenx != len(Y):
        raise ValueError
    lenv = int(lenx * ratio) # lent = lenx - lenv
    indices = np.arange(lenx)
    np.random.shuffle(indices)
    x_val = X[indices[:lenv]]
    y_val = Y[indices[:lenv]]
    x_train = X[indices[lenv:]]
    y_train = Y[indices[lenv:]]
    return (x_train, y_train, x_val, y_val)


def make_fat_lrs(slim_lr, fat=4):

    n,h,w,c = slim_lr.shape
    if (h!=32 or w!=32):
        raise ValueError
    index = np.arange(n)
    in1 = index * 3 
    in2 = index * 3 + 1
    in3 = index * 3 + 2

    np.random.shuffle(in1)
    np.random.shuffle(in2)
    np.random.shuffle(in3)

    while len(in1) >= fat:
        for i in range(fat):
            if in1[i] == 0:
                r1 = 0
            else :
                r1 = int(in1[i]/3)
            slr_temp = slim_lr[r1,:,:,0]
            slr_temp = np.expand_dims(slr_temp, axis=0)
            slr_temp = np.expand_dims(slr_temp, axis=3)
            if i==0:
                slr = slr_temp
            else:
                slr = np.concatenate((slr, slr_temp), axis=3)
        
        for i in range(fat):
            r1 = int(in2[i]/c)
            slr_temp = slim_lr[r1,:,:,1]
            slr_temp = np.expand_dims(slr_temp, axis=0)
            slr_temp = np.expand_dims(slr_temp, axis=3)
            slr = np.concatenate((slr, slr_temp), axis=3)

        for i in range(fat):
            r1 = int(in2[i]/c)
            slr_temp = slim_lr[r1,:,:,2]
            slr_temp = np.expand_dims(slr_temp, axis=0)
            slr_temp = np.expand_dims(slr_temp, axis=3)
            slr = np.concatenate((slr, slr_temp), axis=3)

        if len(in1) == n:
            fat_lr = slr
        else:
            fat_lr = np.concatenate((fat_lr, slr), axis=0)
        for i in range(fat):
            in1 = np.delete(in1, [0])
            in2 = np.delete(in2, [0])
            in3 = np.delete(in3, [0])
            
    return fat_lr


def fat_lr_bg(slim_lr, fat=4, batch_size=32):

    while True:
        yield make_fat_lrs_simple(slim_lr, fat = fat, batch_size = batch_size)


def make_fat_lrs_simple(slim_lr, fat = 4, batch_size = 32):

    n,h,w,c = slim_lr.shape

    if (h!=32 or w!=32):
        raise ValueError

    index = np.arange(n)
    np.random.shuffle(index)
    del_list = np.arange(fat)

    for j in range(batch_size):
        k=0
        for i in index[:fat]:

            slr_temp = slim_lr[i,:,:,:]                         # 인덱스 이미지 한장 온전히 뽑아냄 1,32,32,3
            slr_temp = np.expand_dims(slr_temp, axis=0)         # add axis

            if k == 0:
                slr = slr_temp
            else:
                slr = np.concatenate((slr,slr_temp), axis=3)    # fat 반복문 다돌면 1,32,32,12
            k = k + 1

        if j == 0:
            fat_lr = slr
        else:
            fat_lr = np.concatenate((fat_lr,slr), axis=0)       # four dimension (32,32,32,12)
        index = np.delete(index, del_list) # 앞에 4개 삭제

    return fat_lr


def make_fat_lrs_del(slim_lr, fat=4):
    n,h,w,c = slim_lr.shape
    if (h!=32 or w!=32):
        raise ValueError
    index = np.arange(n*c)
    np.random.shuffle(index)
    while len(index) >= fat*c:
        for i in range(fat*c):
            r1 = int(index[i]/c)
            r2 = index[i]%c
            slr_temp = slim_lr[r1,:,:,r2]
            slr_temp = np.expand_dims(slr_temp, axis=0)
            slr_temp = np.expand_dims(slr_temp, axis=3)
            if i==0:
                slr = slr_temp
            else:
                slr = np.concatenate((slr, slr_temp), axis=3)
        if len(index) == n*c:
            fat_lr = slr
        else:
            fat_lr = np.concatenate((fat_lr, slr), axis=0)
        for i in range(fat*c):
            index = np.delete(index, [0])
    return fat_lr



if __name__ == '__main__':

    test = np.load('./CP_LR_T/lr_image_t_1.npy')
    a = make_fat_lrs_simple(test)
    print(a.shape)
    print(test.shape)
