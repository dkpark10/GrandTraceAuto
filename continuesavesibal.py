import os,sys

if __name__ == '__main__':
    
    path = '/home/bit/final_cp_dataset/ver5/ALL/LR/'
    what = sys.argv[1]
    num = sys.argv[2]
    num = int(num)

    path += what
    
    pathlist = os.listdir(path)

    for idx in range(len(pathlist)):

        orig = path + "/" + pathlist[idx]
        chg = path + "/" + what[-1] + "_" + str(num) + ".png"
        num += 1 
        os.rename(orig,chg)


