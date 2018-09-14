import IndianPines_Input_DFC
import time
from collections import Counter
import numpy as np

def make_hparam_string(patch_size):
    return "ps=%d" % patch_size


# Input data
print("------------------------")
print("Input data")
print("------------------------")
input = IndianPines_Input_DFC.IndianPines_Input()
print("Classified pixels", np.count_nonzero(input.complete_gt))
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


file = open("resultados2.txt", "w+")


for patch_size in [9]:#[1,3,5,9,15,21,25,31]:


    print("Patch size:" + str(patch_size))
    config['patch_size'] = patch_size
    log_dir = "/home/manuel/resultados/ps_" + str(patch_size) + "/"
    config['log_dir'] = log_dir + make_hparam_string(config['patch_size'])

    a = time.time()
    X_train, y_train, X_test, y_test = input.read_data(config['patch_size'])

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
#     save_path, final_test_acc,  conf_matrix = CNNTrain_2D.train_model(X_train, y_train, X_test, y_test, config)
#
#     print("Test accuracy: ", final_test_acc*100)
#     file.write("Final test accuracy: %.3f" % final_test_acc + "\n" )
#
#     # Clear memory
#     del X_train, X_test, y_train, y_test


file.close()

# raw, accuracy = Decoder_DFC.decode(input, config, save_path)
# plt.imshow(raw)
# envi.save_image("ip3.hdr", raw, dtype=int)
































