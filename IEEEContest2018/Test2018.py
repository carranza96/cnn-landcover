from IEEEContest2018 import Input2018
from IEEEContest2018 import Decoder2018 as Decoder
import time
from collections import Counter
import numpy as np
import CNNTrain_2D
import spectral.io.envi as envi
from spectral import imshow
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import EditedNearestNeighbours,NearMiss,RandomUnderSampler


def make_hparam_string(patch_size):
    return "ps%d" % patch_size


# Input data
print("------------------------")
print("Input data")
print("------------------------")
input = Input2018.Input2018()
print("Training pixels", np.count_nonzero(input.train_data))
print("------------------------")


# Configurable parameters
config = {}
config['patch_size'] = 7
config['in_channels'] = input.bands
config['num_classes'] = input.num_classes
config['kernel_size'] = 3
config['conv1_channels'] = 32
config['conv2_channels'] = 64
config['fc1_units'] = 1024
config['batch_size'] = 16
config['max_epochs'] = 60
config['train_dropout'] = 0.8
config['initial_learning_rate'] = 0.01
config['decaying_lr'] = True
config['seed'] = None
folder = 'IEEEContest2018/'
undersampling = True
oversampling = False
rotation_oversampling = True
validation_set = False




file = open(folder + "resultados.txt", "w+")

for patch_size in [15]:

    print("Patch size:" + str(patch_size))
    config['patch_size'] = patch_size
    log_dir = folder + "resultados/ps" + str(patch_size) + "/"
    config['log_dir'] = log_dir

    a = time.time()

    # X, y = input.read_train_data(config['patch_size'])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=37, stratify=y)

    #X_test, y_test, X_train, y_train = input.read_data(config['patch_size'])
    # del X, y

    # X_train, y_train = input.read_train_data(config['patch_size'])


    if undersampling:
        X, y = input.read_data(patch_size=1)
        X_reshaped = X.reshape(X.shape[0], input.bands)
        print("Elements before undersampling: %i" % len(X))
        print(sorted(Counter(y).items()))
        # enn = RandomUnderSampler(sampling_strategy={0:10000,1:15000,3:10000,4:10000,5:10000,7:15000,8:20000,9:10000, 10:10000, 12:10000,13:10000,
        #                                             14:10000,15:10000,17:10000,18:10000,19:10000},random_state=0)
        enn = RandomUnderSampler(
            sampling_strategy={0: 8000, 1: 8000, 3: 6000, 4: 6000, 5: 6000, 7: 8000, 8: 7000, 9: 6000,
                               10: 6000, 12: 6000, 13: 6000,
                               14: 6000, 15: 6000, 17: 6000, 18: 6000, 19: 6000}, random_state=37)

        enn.fit_resample(X_reshaped, y)
        print("Elements after undersampling: %i" % len(enn.sample_indices_))
        X_train, y_train, X_test, y_test = input.train_test_data(patch_size=config['patch_size'], train_indices = enn.sample_indices_)
        # X_test, y_test = np.delete(X_train, enn.sample_indices_, axis=0), np.delete(y_train, enn.sample_indices_, axis=0)
        # X_train, y_train = np.take(X_train, enn.sample_indices_, axis=0), np.take(y_train, enn.sample_indices_, axis=0)
        print(sorted(Counter(y_train).items()))


    else:
        X_train, y_train = input.read_data(config['patch_size'])

    if validation_set:
        X_train, X_val, y_train, y_val = \
            train_test_split(X_train, y_train, test_size=0.5, random_state=42, stratify=y_train)


    if oversampling:
        X_train, y_train = input.oversample_data(X_train, y_train, patch_size)


    if rotation_oversampling:
        X_train, y_train = input.rotation_oversampling(X_train, y_train)




    print('Start training')


    print(time.time() - a)

    file.write("\n--------------------------------\n")
    file.write("Size training set: %d\n" %len(X_train))
    print("Size training set", len(X_train))
    if validation_set:
        file.write("Size validation set: %d\n" % len(X_val))
        print("Size validation set", len(X_val))
    file.write("Size test set: %d\n" %len(X_test))
    print("Size test set", len(X_test))


    file.write("\n------------------\nResults for patch size " + str(patch_size) + ":\n")


    if validation_set:
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

    else:
        file.write("Class distribution:\n")
        file.write("#;Train;Test\n")
        dtrain = Counter(y_train)
        dtest = Counter(y_test)
        for i in range(input.num_classes):
            file.write(str(i + 1) + ";" + str(dtrain[i]) + ";" + str(dtest[i]) + "\n")

        save_path, test_acc, conf_matrix = CNNTrain_2D.train_model(X_train, y_train, X_test, y_test, config)

        # Clear memory
        del X_train, X_test, y_train, y_test


    train_acc = 0



    # Decode result
    raw = Decoder.decode(input, config, save_path)

    print("Train accuracy: ", train_acc)
    file.write("Train accuracy; %.3f" % train_acc + "\n")
    if validation_set:
        print("Validation accuracy: ", val_acc)
        file.write("Validation accuracy; %.3f" % val_acc + "\n")
    print("Test accuracy: ", test_acc)
    file.write("Test accuracy; %.3f" % test_acc + "\n")
    conf_matrix.to_dataframe().to_csv(config['log_dir'] + "conf_matrix" + str(patch_size))
    conf_matrix.classification_report.to_csv(config['log_dir'] + "classification_report"
                                             + str(patch_size))



    # Output image
    envi.save_image(config['log_dir'] + "ps" + str(patch_size) + ".hdr",
                    raw, dtype='uint8', force=True, interleave='BSQ', ext='raw')


    output = Decoder.output_image(input, raw)
    # view = imshow(output)
    # plt.savefig(config['log_dir'] + 'img/' + str(patch_size) +'.png')


    # Image with legend
    labelPatches = [patches.Patch(color=input.color_scale.colorTics[x + 1] / 255., label=input.class_names[x]) for x in
                    range(input.num_classes)]
    fig = plt.figure(2)
    lgd = plt.legend(handles=labelPatches, ncol=1, fontsize='small', loc=2, bbox_to_anchor=(1, 1))
    imshow(output, fignum=2)
    # fig.savefig(config['log_dir'] + 'img/' + str(patch_size) + '_lgd.png',
    # bbox_extra_artists=(lgd,), bbox_inches='tight')


    #save_rgb('ps'+str(patch_size)+'.png', output, format='png')



file.close()


































