from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from IndianPines import IndianPines_Input
import time
import numpy as np
from collections import Counter
from IndianPines import IndianPines_CV_DecoderAux
# import Test_Split
from spectral import imshow, save_rgb
import matplotlib.pyplot as plt
from IndianPines import IndianPines_CV_Postprocessing
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# Input data
input = IndianPines_Input.IndianPines_Input()

# Configurable parameters
patch_size = 3
seed = None
folder = "IndianPines/CV_SVM_c50_G01_NoPost/"
rotation_oversampling = False
feature_selection = False

cv_reports = []

for i in range(1):

    reports = []

    print("Patch size:" + str(patch_size))
    log_dir = folder + str(i) + "/"
    directory = os.path.dirname(log_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file = open(log_dir + "cv" + str(i) + ".csv", "w+")

    X, y = input.read_data(patch_size)
    X = X.reshape(len(X), -1)



    skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=i)

    file.write("\n------------------\nResults for patch size " + str(patch_size) + ":\n")
    fold_num = 1

    accuracies = []
    final_test_acc = 0

    for train_index, test_index in skfold.split(X, y):

        fold_log_dir = log_dir + "f=%d/" % (fold_num)
        directory = os.path.dirname(fold_log_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        print('Start training')
        a = time.time()
        # X_train, X_test = np.take(X, train_index, axis=0), np.take(X, test_index, axis=0)
        # print(time.time() - a)
        # y_train, y_test = np.take(y, train_index, axis=0), np.take(y, test_index, axis=0)

        X_train, y_train, X_test, y_test = input.read_train_test_data(patch_size)
        X_train = X_train.reshape(len(X_train), -1)
        X_test = X_test.reshape(len(X_test),-1)


        img_train, img_test = input.train_test_images(train_index, test_index)
        save_rgb(fold_log_dir + "train.png", img_train, format='png')
        save_rgb(fold_log_dir + "test.png", img_test, format='png')

        if feature_selection:
            fs = ExtraTreesClassifier(n_estimators=100)
            fs = fs.fit(X_train, y_train)
            model = SelectFromModel(fs, prefit=True)
            X_train, X_test = model.transform(X_train), model.transform(X_test)
            print(X_train.shape)
        else:
            model = None


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

        # clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=None, n_jobs=-1)
        clf = svm.SVC(C=50, gamma=0.1, kernel='rbf')
        # clf = KNeighborsClassifier(n_neighbors=5)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print("Train accuracy:" + str(accuracy_score(y_train, clf.predict(X_train))))
        print("Test accuracy:" + str(acc))

        raw, train_acc, test_acc = IndianPines_CV_DecoderAux.decode(input, patch_size, train_index, test_index, clf, feature_selector=model)

        # filt_img, post_test_acc = IndianPines_CV_Postprocessing.apply_modal_filter(input, raw, train_index, test_index)
        filt_img, post_test_acc = IndianPines_CV_Postprocessing.clean_image(input, raw), test_acc

        conf_matrix = IndianPines_CV_Postprocessing.get_conf_matrix(input, filt_img, train_index, test_index)

        accuracies.append(post_test_acc)

        fold_oa = conf_matrix.stats_overall['Accuracy']
        fold_kappa = conf_matrix.stats_overall['Kappa']
        reports.append(conf_matrix.classification_report)

        file.write(str(fold_num) + ";" + "%.3f;" % train_acc + "%.3f;" % test_acc + "%.3f;" % post_test_acc + "%.3f" % fold_kappa +"\n")

        conf_matrix.classification_report.to_csv(fold_log_dir + "classification_report" + str(fold_num))

        save_rgb(fold_log_dir + "outmap.png", filt_img, color_scale=input.color_scale, format='png')

        # Clear memory
        del X_train, X_test, y_train, y_test

        fold_num += 1

    concat_reports = pd.concat(reports).astype('float')
    avg_reports = concat_reports.groupby(concat_reports.index).mean()
    avg_reports.to_csv(log_dir.split("f")[0] + "avg_report.csv")
    std_reports = concat_reports.groupby(concat_reports.index).std()
    std_reports.to_csv(log_dir.split("f")[0] + "avg_report_std.csv")

    print(accuracies)
    mean_acc = np.asarray(accuracies, dtype=float).mean()
    file.write("Mean accuracy;" + "%.3f" % mean_acc + "\n\n")

    file.close()

cv_reports.append(avg_reports)

concat_cv_reports = pd.concat(cv_reports).astype('float')
avg_cv_reports = concat_cv_reports.groupby(concat_cv_reports.index).mean()
avg_cv_reports.to_csv(folder + "10x5cv_report.csv")
std_cv_reports = concat_cv_reports.groupby(concat_cv_reports.index).std()
std_cv_reports.to_csv(folder + "10x5cv_report_std.csv")


