from IndianPines import IndianPines_Input
import Decoder
import time
from collections import Counter
import numpy as np
import CNNTrain_2D
import spectral.io.envi as envi
from spectral import imshow
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def make_hparam_string(patch_size):
    return "ps%d" % patch_size


# Input data
print("------------------------")
print("Input data")
print("------------------------")
input = IndianPines_Input.IndianPines_Input()
print("Training pixels", np.count_nonzero(input.train_data))
print("Test pixels", np.count_nonzero(input.test_data))
print("------------------------")


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


for patch_size in [3,5,7,9]:#[1,3,5,9,15,21,25,31]:


    print("Patch size:" + str(patch_size))
    config['patch_size'] = patch_size
    log_dir = "resultados/ps_" + str(patch_size) + "/"
    config['log_dir'] = log_dir + make_hparam_string(config['patch_size'])

    a = time.time()
    X_train, y_train, X_test, y_test = input.read_data(config['patch_size'], oversampling=False)

    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=0.5, random_state=42, stratify=y_train)

    print('Start training')

    print(time.time() - a)

    file.write("\n--------------------------------\n")
    file.write("Size training set: %d\n" %len(X_train))
    file.write("Size validation set: %d\n" % len(X_val))
    file.write("Size test set: %d\n" %len(X_test))
    print("Size training set", len(X_train))
    print("Size validation set", len(X_val))
    print("Size test set", len(X_test))


    file.write("\n------------------\nResults for patch size " + str(patch_size) + ":\n")


    file.write("Class distribution:\n")
    file.write("#;Train;Validation;Test\n")
    dtrain = Counter(y_train)
    dval = Counter(y_val)
    dtest = Counter(y_test)
    for i in range(input.num_classes):
        file.write(str(i+1)+";" + str(dtrain[i]) + ";" + str(dval[i]) + ";" + str(dtest[i]) + "\n")
#
    save_path, val_acc,  conf_matrix = CNNTrain_2D.train_model(X_train, y_train, X_val, y_val, config)
    # save_path, final_test_acc,  conf_matrix = CNNTrain_2D.train_model(X_test, y_test, X_train, y_train, config)


    # Clear memory
    del X_train, X_val, X_test, y_train, y_val, y_test

    raw, train_acc, test_acc = Decoder.decode(input, config, save_path)

    print("Train accuracy: ", train_acc)
    print("Validation accuracy: ", val_acc)
    print("Test accuracy: ", test_acc)

    file.write("Train accuracy; %.3f" % train_acc + "\n")
    file.write("Validation accuracy; %.3f" % val_acc + "\n" )
    file.write("Test accuracy; %.3f" % test_acc + "\n")

    # Output image
    envi.save_image(config['log_dir'] + ".hdr", raw, dtype=int, force=True)
    output = Decoder.output_image(input, raw)
    view = imshow(output)
    plt.savefig('ps'+str(patch_size)+'.png')
    #save_rgb('ps'+str(patch_size)+'.png', output, format='png')



file.close()


































