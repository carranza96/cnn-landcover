from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold

import CNNTrain_2D
from IndianPines import IndianPines_Input
import time
import numpy as np
from collections import Counter
from IndianPines import IndianPines_CV_Decoder
#import Test_Split
from spectral import imshow
import matplotlib.pyplot as plt


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

def make_hparam_string(fold_num):
    return "f=%d" % (fold_num)


# Input data
input = IndianPines_Input.IndianPines_Input()


# Configurable parameters
config = {}
#config['patch_size'] = 15
config['kernel_size'] = 3
config['in_channels'] = input.bands
config['num_classes'] = input.num_classes
config['conv1_channels'] = 32
config['conv2_channels'] = 64
config['fc1_units'] = 1024
config['batch_size'] = 16
config['max_epochs'] = 1
config['train_dropout'] = 0.5
config['initial_learning_rate'] = 0.01
config['decaying_lr'] = True
config['seed'] = None


rotation_oversampling = False






for i in range(1,2):

    #[1,3,5,9,15,21,25,31]
    for patch_size in [5]:#[1,3,5,9,15,21,25,31]:

        print("Patch size:" + str(patch_size))
        config['patch_size'] = patch_size
        log_dir = "IndianPines/CV/ps" + str(patch_size) + "/" + str(i) + "/"
        config['log_dir'] = log_dir

        file = open("IndianPines/CV/cv" + str(i) + ".txt", "w+")

        X, y = input.read_data(config['patch_size'])
        skfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=i)
        # skfold = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=37)
        #folds = Test_Split.train_test_split(X,y)

        file.write("\n------------------\nResults for patch size " + str(patch_size) + ":\n")
        fold_num = 1

        accuracies = []
        final_test_acc = 0

        for train_index, test_index in skfold.split(X,y):

            config['log_dir'] = log_dir + "f=%d/" % (fold_num)

            print('Start training')
            a = time.time()
            X_train, X_test = np.take(X, train_index, axis=0), np.take(X, test_index, axis=0)
            print(time.time() - a)
            y_train, y_test = np.take(y, train_index, axis=0), np.take(y, test_index, axis=0)


            img_train, img_test = input.train_test_images(train_index,test_index)
            imshow(img_train)
            imshow(img_test)



            if rotation_oversampling:
                X_train, y_train = input.rotation_oversampling(X_train, y_train)

            print("Size training set", len(X_train))
            print("Size test set", len(X_test))

            if fold_num==1:
                file.write("Size training set: %d\n" % len(X_train))
                file.write("Size test set: %d\n" % len(X_test))
                file.write("Class distribution:\n")
                file.write("Train;Test\n")
                dtrain = Counter(y_train)
                dtest = Counter(y_test)
                for i in range(input.num_classes):
                    file.write(str(i)+";"+ str(dtrain[i]) + ";" + str(dtest[i]) + "\n")

            save_path, test_acc, conf_matrix = CNNTrain_2D.train_model(X_train, y_train, X_test, y_test, config)

            print(test_acc)

            raw, train_acc, test_acc = IndianPines_CV_Decoder.decode(input, config, train_index, test_index, save_path)

            print("Train", train_acc)
            print(test_acc)
            accuracies.append(final_test_acc)
            file.write("Fold "+ str(fold_num) + ";" + "%.3f" % test_acc + "%.3f" % test_acc + "\n")


            conf_matrix.classification_report.to_csv(config['log_dir'] + "classification_report"
                                                     + str(fold_num))

            # conf_matrix.plot()
            # conf_matrix.print_stats()

            # Clear memory
            del X_train, X_test, y_train, y_test

            fold_num += 1
            break


        print(accuracies)
        mean_acc = np.asarray(accuracies,dtype=float).mean()
        file.write("Mean accuracy;" + "%.3f" % mean_acc + "\n\n" )



        file.close()





























