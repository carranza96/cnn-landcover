from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from IndianPines import IndianPines_Input
from Pavia import Pavia_Input
from Flevoland import Flevoland_Input
from SanFrancisco import SanFrancisco_Input
from Salinas import Salinas_Input
import time
import numpy as np
from collections import Counter
from spectral import imshow, save_rgb
import CV_Decoder,CV_Postprocessing
import os
import pandas as pd
import CNNTrain_2D


# Input data
images = ["IndianPines", "Pavia", "Flevoland", "SanFrancisco", "Salinas"]
images_inputs = {"IndianPines": IndianPines_Input.IndianPines_Input(),
                 "Pavia": Pavia_Input.Pavia_Input(),
                 "Flevoland": Flevoland_Input.Flevoland_Input(),
                 "SanFrancisco": SanFrancisco_Input.SanFrancisco_Input(),
                 "Salinas": Salinas_Input.Salinas_Input()
                 }

# Select image to test
selected_img = images[0]
input = images_inputs[selected_img]

print("Image:" + selected_img)

patch_size = 5

config = {}
config['patch_size'] = patch_size
config['kernel_size'] = 3
config['in_channels'] = input.bands
config['num_classes'] = input.num_classes
config['conv1_channels'] = 32
config['conv2_channels'] = 64
config['fc1_units'] = 1024
config['batch_size'] = 16
config['max_epochs'] = 50
config['train_dropout'] = 0.8
config['initial_learning_rate'] = 0.01
config['decaying_lr'] = True
config['seed'] = None
folder = selected_img + "/CV_CNN54ROT_80epochs/"
rotation_oversampling = True
apply_filter = False



# 5 partitions to the dataset
X, y, positions = input.read_data(patch_size)
dataset_reduction = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

partition = 1
overall_reports = []
for discarded_indices, selected_indices in dataset_reduction.split(X, y):


    print("Patch size:" + str(patch_size))
    partition_dir = folder + "Partition" + str(partition) + "/"
    directory = os.path.dirname(partition_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    X_reduced, y_reduced = np.take(X, selected_indices, axis=0), np.take(y, selected_indices, axis=0)
    selected_positions = np.take(positions, selected_indices, axis=0)

    partition_reports = []

    # Repeat CV process 5 times
    for cv_index in range(5):

        cv_dir = partition_dir + "cv" + str(cv_index) + "/"
        directory = os.path.dirname(cv_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        file = open(cv_dir + "cv" + str(cv_index) + ".csv", "w+")

        skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=cv_index)

        file.write("\n------------------\nResults for patch size " + str(patch_size) + ":\n")
        fold_num = 1

        cv_accuracies = []
        cv_reports = []

        for train_index, test_index in skfold.split(X_reduced, y_reduced):

            fold_dir = cv_dir + "f=%d/" % (fold_num)
            directory = os.path.dirname(fold_dir)
            if not os.path.exists(directory):
                os.makedirs(directory)
            config['log_dir'] = fold_dir

            print('Start training')
            a = time.time()
            X_train, X_test = np.take(X_reduced, train_index, axis=0), np.take(X_reduced, test_index, axis=0)
            print(time.time() - a)
            y_train, y_test = np.take(y_reduced, train_index, axis=0), np.take(y_reduced, test_index, axis=0)

            train_positions = np.take(selected_positions, train_index, axis=0)
            test_positions = np.take(selected_positions, test_index, axis=0)

            img_train, img_test = input.train_test_images(train_positions, test_positions)
            save_rgb(fold_dir + "train.png", img_train, format='png')
            save_rgb(fold_dir + "test.png", img_test, format='png')


            if rotation_oversampling:
                X_train, y_train = input.rotation_oversampling(X_train, y_train)

            print("Size training set", len(X_train))
            print("Size test set", len(X_test))

            if fold_num == 1:
                file.write("Size training set: %d\n" % len(X_train))
                file.write("Size test set: %d\n" % len(X_test))
                file.write("Class distribution:\n")
                file.write("Train;Test\n")
                dtrain = Counter(y_train)
                dtest = Counter(y_test)
                for i in range(input.num_classes):
                    file.write(str(i) + ";" + str(dtrain[i]) + ";" + str(dtest[i]) + "\n")
                file.write("Fold;Train acc; Test acc; Test Post acc;Kappa\n")

            print('Start training')
            t = time.time()

            save_path, test_acc, _ = CNNTrain_2D.train_model(X_train, y_train, X_test, y_test, config)

            t = time.time() - t
            print(t)
            if fold_num == 1:
                file.write("Time %0.3f\n" % t)

            print("Test accuracy:" + str(test_acc))

            raw, train_acc, test_acc = CV_Decoder.decode_cnn(input, config, train_positions, test_positions, save_path)

            print("Train accuracy:" + str(train_acc))
            print("Test accuracy:" + str(test_acc))


            save_rgb(fold_dir + "outmap.png", raw, color_scale=input.color_scale, format='png')


            if apply_filter:
                filt_img, post_test_acc = CV_Postprocessing.apply_modal_filter(input, raw, train_positions, test_positions)
            else:
                filt_img, post_test_acc = CV_Postprocessing.clean_image(input, raw), test_acc


            conf_matrix = CV_Postprocessing.get_conf_matrix(input, filt_img, test_positions)


            fold_oa = conf_matrix.stats_overall['Accuracy']
            fold_kappa = conf_matrix.stats_overall['Kappa']

            cv_accuracies.append(post_test_acc)
            cv_reports.append(conf_matrix.classification_report)

            file.write(str(fold_num) + ";" + "%.3f;" % train_acc + "%.3f;" % test_acc + "%.3f;" % post_test_acc + "%.3f" % fold_kappa +"\n")

            conf_matrix.classification_report.to_csv(fold_dir + "classification_report" + str(fold_num))

            save_rgb(fold_dir + "outmap_postprocessing.png", filt_img, color_scale=input.color_scale, format='png')

            # Clear memory
            del X_train, X_test, y_train, y_test

            fold_num += 1

        concat_cv_reports = pd.concat(cv_reports).astype('float')
        avg_cv_reports = concat_cv_reports.groupby(concat_cv_reports.index).mean()
        avg_cv_reports.to_csv(cv_dir + "avg_cv" + str(cv_index) + "_report.csv")
        std_cv_reports = concat_cv_reports.groupby(concat_cv_reports.index).std()
        std_cv_reports.to_csv(cv_dir + "std_cv"+str(cv_index)+ "_report.csv")

        print(cv_accuracies)
        mean_acc = np.asarray(cv_accuracies, dtype=float).mean()
        file.write("Mean accuracy;" + "%.3f" % mean_acc + "\n\n")

        file.close()

        partition_reports.append(avg_cv_reports)


    concat_partition_reports = pd.concat(partition_reports).astype('float')
    avg_partition_reports = concat_partition_reports.groupby(concat_partition_reports.index).mean()
    avg_partition_reports.to_csv(partition_dir + "part" + str(partition) + "_5x3cv_report.csv")
    std_partition_reports = concat_partition_reports.groupby(concat_partition_reports.index).std()
    std_partition_reports.to_csv(partition_dir + "part" + str(partition) + "_5x3cv_report_std.csv")
    partition += 1

    overall_reports.append(avg_partition_reports)

concat_overall_reports = pd.concat(overall_reports).astype('float')
avg_overall_reports = concat_overall_reports.groupby(concat_overall_reports.index).mean()
avg_overall_reports.to_csv(folder + "5x3cv_overall_report")
std_overal_reports = concat_overall_reports.groupby(concat_overall_reports.index).std()
std_partition_reports.to_csv(folder + "5x3cv_overall_report_std")

