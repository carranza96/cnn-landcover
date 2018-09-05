from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

import CNNTrain_2D
import Salinas_Input
import time
import numpy as np
from collections import Counter


# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('log_dir', '/tmp/indian_pines', """Directory where to write event logs and checkpoint.""")
# flags.DEFINE_integer('patch_size', 3, 'Size of input patch image')
# flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# flags.DEFINE_integer('max_epochs', 1, 'Number of steps to run trainer.')
# flags.DEFINE_integer('conv1_channels', 500, 'Number of filters in convolutional layer 1.')
# flags.DEFINE_integer('conv2_channels', 100, 'Number of filters in convolutional layer 2.')
# flags.DEFINE_integer('fc1_units', 200, 'Number of units in dense layer 1.')
# flags.DEFINE_float('train_dropout', 0.5, 'Probability of dropping out units.')
# flags.DEFINE_integer('batch_size', 100, 'Batch size.')

def make_hparam_string(patch_size,learning_rate, learning_rate_decay,fold_num):
    lr_decay = "lr_d=Y" if learning_rate_decay else "lr_d=N"
    return "ps=%d,lr_%.0E,%s,f=%d" % (patch_size,learning_rate,lr_decay,fold_num)


# Configurable parameters
config = {}
#config['patch_size'] = 15
config['kernel_size'] = 3
config['conv1_channels'] = 32
config['conv2_channels'] = 64
config['fc1_units'] = 1024
config['batch_size'] = 16
config['max_epochs'] = 100
config['train_dropout'] = 0.5
config['initial_learning_rate'] = 0.01
config['decaying_lr'] = True
log_dir = '/home/manuel/Escritorio/indianpines/'
config['log_dir'] = log_dir







# Input data
input = Salinas_Input.Salinas_Input()
# Force input pipeline to CPU:0 to avoid operations sometimes ending up on
# GPU and resulting in a slow down.
# with tf.device('/cpu:0'):




file = open("cv.txt","w+")

#[1,3,5,9,15,21,25,31]
for patch_size in [9]:#[1,3,5,9,15,21,25,31]:

    print("Patch size:" + str(patch_size))
    config['patch_size'] = patch_size
    log_dir = "/home/manuel/resultados/cv" + str(patch_size) + "/"
    config['log_dir'] = log_dir


    X, y = input.read_data(config['patch_size'])
    #skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=41)
    skfold = StratifiedShuffleSplit(n_splits=10, test_size=0.7, random_state=1)


    file.write("\n------------------\nResults for patch size " + str(patch_size) + ":\n")
    fold_num = 1

    accuracies = []

    for train_index, test_index in skfold.split(X,y):

        config['log_dir'] = log_dir + make_hparam_string(config['patch_size'],config['initial_learning_rate'],config['decaying_lr'],fold_num)
        print('Start training')
        a = time.time()
        X_train = np.take(X,train_index,axis=0)
        X_test = np.take(X,test_index,axis=0)


        print(time.time() - a)
        y_train, y_test = np.take(y,train_index,axis=0), np.take(y,test_index,axis=0)
        print("Size training set", len(X_train))
        print("Size test set", len(X_test))

        if fold_num==1:
            file.write("Size training set: %d\n" % len(X_train))
            file.write("Size test set: %d\n" % len(X_test))
            file.write("Class distribution:\n")
            file.write("Train;Test\n")
            dtrain = Counter(y_train)
            dtest = Counter(y_test)
            for i in range(Salinas_Input.NUM_CLASSES):
                file.write(Salinas_Input.CLASS_NAMES[i]+";"+ str(dtrain[i]) + ";" + str(dtest[i]) + "\n")

        save_path,final_test_acc,conf_matrix = CNNTrain_2D.train_model(X_train, y_train, X_test, y_test, config)

        print(final_test_acc)
        accuracies.append(final_test_acc)
        file.write("Fold "+ str(fold_num) + ";" + "%.3f" % final_test_acc + "\n"  )


        # Clear memory
        del X_train, X_test, y_train, y_test

        fold_num+=1


    print(accuracies)
    mean_acc = np.asarray(accuracies,dtype=float).mean()
    file.write("Mean accuracy;" + "%.3f" % mean_acc + "\n\n" )



file.close()

























