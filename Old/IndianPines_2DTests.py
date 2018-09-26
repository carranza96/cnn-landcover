import time
import numpy as np
from Old import Test_Split, IndianPines_Input
import CNNTrain_2D
from collections import Counter



def make_hparam_string(patch_size,learning_rate):
    return "ps=%d,lr_%.0E" % (patch_size,learning_rate)


# Configurable parameters
config = {}
config['patch_size'] = 9
config['kernel_size'] = 3
config['conv1_channels'] = 32
config['conv2_channels'] = 64
config['fc1_units'] = 1024
config['batch_size'] = 16
config['max_epochs'] = 50
config['train_dropout'] = 0.5
config['initial_learning_rate'] = 0.01
config['decaying_lr'] = True



# Input data
input = IndianPines_Input.IndianPines_Input()
# Force input pipeline to CPU:0 to avoid operations sometimes ending up on
# GPU and resulting in a slow down.
# with tf.device('/cpu:0'):

file = open("resultados.txt", "w+")

for patch_size in [7]:#[1,3,5,9,15,21,25,31]:


    print("Patch size:" + str(patch_size))
    config['patch_size'] = patch_size
    log_dir = "/home/manuel/resultados/ps_" + str(patch_size) + "/"
    config['log_dir'] = log_dir + make_hparam_string(config['patch_size'],
                                                     config['initial_learning_rate'])

    X, y = input.read_data(config['patch_size'])


    print('Start training')
    train_index,test_index = Test_Split.train_test_split(X, y)
    a = time.time()
    X_train, X_test = np.take(X, train_index, axis=0), np.take(X, test_index, axis=0)
    y_train, y_test = np.take(y,train_index,axis=0), np.take(y,test_index,axis=0)
    print(time.time() - a)
    file.write("Size training set: %d\n" %len(X_train))
    file.write("Size test set: %d\n" %len(X_test))
    print("Size training set", len(X_train))
    print("Size test set", len(X_test))
    print(y_test.shape)
    print(y_test)

    file.write("\n------------------\nResults for patch size " + str(patch_size) + ":\n")


    file.write("Class distribution:\n")
    file.write("Train;Test\n")
    dtrain = Counter(y_train)
    dtest = Counter(y_test)
    for i in range(IndianPines_Input.NUM_CLASSES):
        file.write(str(i)+";"+ str(dtrain[i]) + ";" + str(dtest[i]) + "\n")

    save_path,final_test_acc,conf_matrix = CNNTrain_2D.train_model(X_train, y_train, X_test, y_test, config)


    print("Test accuracy: ",final_test_acc*100)
    file.write("Final test accuracy: %.3f" % final_test_acc + "\n"  )
    file.write("Confusion matrix:\n")


    conf_matrix.plot()
    conf_matrix.print_stats()

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


































