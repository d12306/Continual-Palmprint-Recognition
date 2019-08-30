import os
import pickle
import numpy as np
from PIL import Image


#im = Image.open('/Users/michael/test.jpg')
'''
permutations = []
for i in range(args.n_tasks):
    indices = np.random.permutation(784)
    print(x_tr[:, indices].shape)#(55000, 784)
    permutations.append((x_tr[:, indices], y_tr, x_val[:, indices], y_val, x_te[:, indices], y_te))
f = open(args.o, "wb")
pickle.dump(permutations, f)
f.close()

'''


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

index = os.listdir('./Green')
index.sort()

hhh = list( range(0,10) )
pick_file=[]
for i in hhh:
    temp = index[i*50: i*50+50]
    train = []
    train_y = []
    validation = []
    validation_y = []
    test = []
    test_y =[]
    #print(temp)
    for item in temp:
    
        file = os.path.join('./Green',item)
        temp1 = os.listdir(str(file))
        timer = 0
        for j in temp1:
            a=np.asarray(Image.open(os.path.join(file,j)).convert('RGB'))
            a = (a-np.mean(a))/np.std(a)
            if timer<6:
                train.append(a)
                train_y.append(int(item)-1-50*i)
                timer+=1
                
            elif timer>=6 and timer <=8:
                test.append(a)
                test_y.append(int(item)-1-50*i)
                timer+=1
            else:
                validation.append(a)
                validation_y.append(int(item)-1-50*i)
                timer+=1
    print(np.asarray(train).shape)
    print(np.asarray(train_y).shape)
    print(np.asarray(validation).shape)
    print(np.asarray(validation_y).shape)
    print(np.asarray(test).shape)
    print(np.asarray(test_y).shape)
    pick_file.append((train, to_categorical(train_y), validation,to_categorical(validation_y), test,to_categorical(test_y)))
f = open('./data.pkl', "wb")
pickle.dump(pick_file, f)
f.close()            
    
        #print()
        #[103.939, 116.779, 123.68]