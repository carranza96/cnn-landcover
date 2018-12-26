from sklearn.model_selection import StratifiedKFold
from IndianPines import IndianPines_Input
from Pavia import Pavia_Input
from Flevoland import Flevoland_Input
from SanFrancisco import SanFrancisco_Input
from Salinas import Salinas_Input
import time
import numpy as np
from collections import Counter
from spectral import save_rgb
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import CV_Postprocessing, CV_Decoder
from pandas_ml import ConfusionMatrix


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



# Configurable parameters
patch_size = 5
seed = None
rotation_oversampling = False
feature_selection = False
apply_filter = False
classifiers = ["RF", "SVM", "1NN", "3NN", "5NN"]
classifier = classifiers[0]
folder = selected_img + "/CV_" + classifier + "/"

if "NN" in classifier:
    feature_selection = True

# 5 partitions to the dataset
X, y, positions = input.read_data(patch_size)
# X = X.reshape(len(X), -1)
dataset_reduction = StratifiedKFold(n_splits=20, shuffle=True, random_state=0)

overall_reports = []
partition = 1

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
    for cv_index in range(1):

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



            X_train, X_test = np.take(X_reduced, train_index, axis=0), np.take(X_reduced, test_index, axis=0)
            y_train, y_test = np.take(y_reduced, train_index, axis=0), np.take(y_reduced, test_index, axis=0)

            train_positions = np.take(selected_positions, train_index, axis=0)
            test_positions = np.take(selected_positions, test_index, axis=0)


            img_train, img_test = input.train_test_images(train_positions, test_positions)
            save_rgb(fold_dir + "train.png", img_train, format='png')
            save_rgb(fold_dir + "test.png", img_test, format='png')


            if rotation_oversampling:
                X_train, y_train = input.rotation_oversampling(X_train, y_train)

            X_train = X_train.reshape(len(X_train), -1)
            X_test = X_test.reshape(len(X_test),-1)

            if feature_selection:
                fs = ExtraTreesClassifier(n_estimators=100)
                fs = fs.fit(X_train, y_train)
                model = SelectFromModel(fs, prefit=True)
                X_train, X_test = model.transform(X_train), model.transform(X_test)
                print(X_train.shape)
            else:
                model = None

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

            if classifier =="RF":
                clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=None, n_jobs=-1)
            elif classifier == "SVM":
                clf = svm.SVC(C=50, gamma=0.01, kernel='rbf')
            elif classifier == "1NN":
                clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
            elif classifier == "3NN":
                clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
            elif classifier == "5NN":
                clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)


            clf.fit(X_train, y_train)

            t = time.time() - t
            print(t)
            if fold_num == 1:
                file.write("Time %0.3f\n" %t)

            y_pred = clf.predict(X_test)

            # train_acc = accuracy_score(y_train, clf.predict(X_train))*100
            train_acc = 0
            test_acc = accuracy_score(y_test, y_pred)*100

            print("Train accuracy:" + str(train_acc))
            print("Test accuracy:" + str(test_acc))


            conf_matrix = ConfusionMatrix(y_test,y_pred)

            # raw = CV_Decoder.decode_sklearn(input, patch_size, clf, feature_selector=model)
            # save_rgb(fold_dir + "outmap.png", raw, color_scale=input.color_scale, format='png')
            #
            #
            # if apply_filter:
            #     filt_img, post_test_acc = CV_Postprocessing.apply_modal_filter(input, raw, train_positions, test_positions)
            # else:
            #     filt_img, post_test_acc = CV_Postprocessing.clean_image(input, raw), test_acc
            #
            # conf_matrix = CV_Postprocessing.get_conf_matrix(input, filt_img, test_positions)

            post_test_acc = test_acc
            fold_oa = conf_matrix.stats_overall['Accuracy']
            fold_kappa = conf_matrix.stats_overall['Kappa']

            cv_accuracies.append(post_test_acc)
            cv_reports.append(conf_matrix.classification_report)

            file.write(str(fold_num) + ";" + "%.3f;" % train_acc + "%.3f;" % test_acc + "%.3f;" % post_test_acc + "%.3f" % fold_kappa +"\n")

            conf_matrix.classification_report.to_csv(fold_dir + "classification_report" + str(fold_num))

            # save_rgb(fold_dir + "outmap_postprocessing.png", filt_img, color_scale=input.color_scale, format='png')

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

