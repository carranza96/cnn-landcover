
import CNNTrain_2D
import IndianPines_Input_DFC
import Decoder_DFC
import tensorflow as tf
import time
import numpy as np
from collections import Counter
#from ConfusionMatrix import plot_confusion_matrix,print_cm,print_confusion_matrix
import matplotlib.pyplot as plt
import spectral.io.envi as envi


def make_hparam_string(patch_size,learning_rate):
    return "ps=%d,lr_%.0E" % (patch_size,learning_rate)


# Input data
input = IndianPines_Input_DFC.IndianPines_Input()

# Configurable parameters
config = {}
config['patch_size'] = 9
config['in_channels'] = input.bands
config['num_classes'] = input.num_classes
config['kernel_size'] = 3
config['conv1_channels'] = 32
config['conv2_channels'] = 64
config['fc1_units'] = 1024
config['batch_size'] = 16
config['max_epochs'] = 80
config['train_dropout'] = 0.5
config['initial_learning_rate'] = 0.01
config['decaying_lr'] = True





file = open("resultados.txt", "w+")

for patch_size in [9]:#[1,3,5,9,15,21,25,31]:


    print("Patch size:" + str(patch_size))
    config['patch_size'] = patch_size
    log_dir = "/home/manuel/resultados/ps_" + str(patch_size) + "/"
    config['log_dir'] = log_dir + make_hparam_string(config['patch_size'],
                                                     config['initial_learning_rate'])

    X_train, y_train = input.read_data(config['patch_size'])




    print('Start training')
    a = time.time()
    X_test,y_test = input.read_data(config['patch_size'])
    print(y_test.shape)

    print(time.time() - a)
    file.write("Size training set: %d\n" %len(X_train))
    file.write("Size test set: %d\n" %len(X_test))
    print("Size training set", len(X_train))
    print("Size test set", len(X_test))


    file.write("\n------------------\nResults for patch size " + str(patch_size) + ":\n")


    file.write("Class distribution:\n")
    file.write("Train;Test\n")
    dtrain = Counter(y_train)
    dtest = Counter(y_test)
    for i in range(input.num_classes):
        file.write(str(i)+";"+ str(dtrain[i]) + ";" + str(dtest[i]) + "\n")

    save_path,final_test_acc,conf_matrix = CNNTrain_2D.train_model(X_train, y_train, X_test, y_test, config)


    print("Test accuracy: ",final_test_acc*100)
    file.write("Final test accuracy: %.3f" % final_test_acc + "\n" )
    file.write("Confusion matrix:\n")


    #.plot()
    #conf_matrix.print_stats()

    # plt.figure()
    # plot_confusion_matrix(conf_matrix, classes=IndianPines_Input.CLASS_NAMES,
    #                       title='Confusion matrix, without normalization')
    # plt.show()
    #
    # print_cm(conf_matrix,IndianPines_Input.CLASS_NAMES)
    # print_confusion_matrix(conf_matrix,IndianPines_Input.CLASS_NAMES)

    # Clear memory
    del X_train, X_test, y_train, y_test


file.close()


raw, accuracy = Decoder_DFC.decode(input,config, save_path)
plt.imshow(raw)
envi.save_image("ip.hdr",raw,dtype=int)
































