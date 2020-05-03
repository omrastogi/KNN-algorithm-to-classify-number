import cv2
import numpy as np
import random


def train():
    number = [0,1,2,3,4,5,6,7,8,9]
    cells=[]
    for label in number:
        print(label)
        directory = "DataSet/"+str(label)  
        training_files = 16
        # training_files+=1
        for i in range(0,training_files):
            # print (i)
            file = directory+"/"+"num"+str(i)+".png"
            digit = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            # show(digit)
            # print ("train size",np.shape(digit))
            cell = digit.flatten()
            # print ("train size",np.size(cell))
            cells.append(cell)
    cells = np.array(cells, dtype=np.float32)        
    k = np.arange(len(number))
    cells_labels = np.repeat(k, training_files)
    # print (cells_labels)
    knn = cv2.ml.KNearest_create()
    knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
    return knn


def show(cells):
    cv2.imshow('img',cells)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test(img,knn):
    # show(img)
    img = img.flatten()
    test_cells = []
    test_cells.append(img)
    test_cells = np.array(test_cells, dtype=np.float32)
    ret, result, neighbours, dist = knn.findNearest(test_cells, k=1) 
    return result[0][0]

# number = ['zero','one','two','three','four','six','seven','eight','nine']
# cells=[]
# cells_labels = []
# for label in number:
#     directory = "DataSet/"+label  
#     for i in range(0,11):
#         file = directory+"/"+"num"+str(i)+".png"
#         digit = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#         cell = digit,flatten()
#         cells.append(cell)
#         cells_labels.append(i)

def efficiency(knn):
    number = [0,1,2,3,4,5,6,7,8,9]
    cells=[]
    total = 0
    correct = 0
    for label in number:
        directory = "DataSet/"+str(label)  
        for i in range(0,15):
            file = directory+"/"+"num"+str(i)+".png"
            # print (file)
            digit = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            total = total + 1
            num =  int(test(digit,knn))
            # print (num,label)
            if int (num) == label:
                correct = correct + 1
    print ("efficiency =",correct/total)




knn = train()
efficiency(knn)
label, i  = 1,9
directory = "DataSet/"+str(label) 
file = directory+"/"+"num"+str(i)+".png"
test_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
num = test(test_img,knn)
# print (int(num))