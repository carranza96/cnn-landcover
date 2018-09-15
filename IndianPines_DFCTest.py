import IndianPines_Input_DFC
import time
from collections import Counter
import numpy as np
import CNNTrain_2D
import Decoder_DFC
import spectral.io.envi as envi
from spectral import imshow


def make_hparam_string(patch_size):
    return "ps=%d" % patch_size


# Input data
print("------------------------")
print("Input data")
print("------------------------")
input = IndianPines_Input_DFC.IndianPines_Input()
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
config['max_epochs'] = 50
config['train_dropout'] = 0.5
config['initial_learning_rate'] = 0.01
config['decaying_lr'] = True


file = open("resultados2.txt", "w+")


for patch_size in [5]:#[1,3,5,9,15,21,25,31]:


    print("Patch size:" + str(patch_size))
    config['patch_size'] = patch_size
    log_dir = "resultados/ps_" + str(patch_size) + "/"
    config['log_dir'] = log_dir + make_hparam_string(config['patch_size'])

    a = time.time()
    X_train, y_train, X_test, y_test = input.read_data(config['patch_size'], oversampling=True)


    print('Start training')

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
        file.write(str(i)+";" + str(dtrain[i]) + ";" + str(dtest[i]) + "\n")
#
    save_path, final_test_acc,  conf_matrix = CNNTrain_2D.train_model(X_train, y_train, X_test, y_test, config)
    # save_path, final_test_acc,  conf_matrix = CNNTrain_2D.train_model(X_test, y_test, X_train, y_train, config)

    print("Test accuracy: ", final_test_acc*100)
    file.write("Final test accuracy: %.3f" % final_test_acc + "\n" )

    # Clear memory
    del X_train, X_test, y_train, y_test


file.close()

raw, train_acc, test_acc = Decoder_DFC.decode(input, config, save_path)
output = Decoder_DFC.output_image(input, raw)
imshow(output)
envi.save_image("ip3.hdr", raw, dtype=int)
































