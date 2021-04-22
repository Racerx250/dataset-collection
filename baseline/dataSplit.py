import os
from shutil import copy

if __name__ == "__main__": 
    trainList = open("train_list.txt", "w")
    valList = open("val_list.txt", "w")
    testList = open("test_list.txt", "w")
    label = -1
    for path, subdirs, files in os.walk('Images'):
        class_size = len(files)
        train_size = round(class_size * 0.7)
        val_size = round(class_size * 0.15) + train_size
        count = 0
        for name in files:
            curpath = os.path.join(path, name)
            curpath = curpath.replace("\\", "/")
            if (count <= train_size) :
                trainList.write(curpath + " " + str(label) + "\n")
                dst = os.path.relpath(path, "Images")
                dst = os.path.join("TrainImages", dst, name)
                dst = dst.replace("\\", "/")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                copy(curpath, dst)
                count += 1
            elif (count > train_size and count <= val_size) :
                valList.write(curpath + " " + str(label) + "\n")
                dst = os.path.relpath(path, "Images")
                dst = os.path.join("ValImages", dst, name)
                dst = dst.replace("\\", "/")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                copy(curpath, dst)
                count += 1
            else :
                testList.write(curpath + " " + str(label) + "\n")
                dst = os.path.relpath(path, "Images")
                dst = os.path.join("TestImages", dst, name)
                dst = dst.replace("\\", "/")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                copy(curpath, dst)
                count += 1
        label += 1
        print(label)

    trainList.close()
    valList.close()
    testList.close()
        
        