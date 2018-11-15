from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from IndianPines import IndianPines_Input
import time
import numpy as np
from collections import Counter
# import Test_Split
from spectral import imshow, save_rgb
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.under_sampling import CondensedNearestNeighbour,EditedNearestNeighbours
import Decoder
from IndianPines import IndianPines_Postprocessing


# Input data
input = IndianPines_Input.IndianPines_Input()

# Configurable parameters
patch_size = 1
seed = None
folder = "IndianPines/Holdout_KNN1/"
rotation_oversampling = False
feature_selection = False
apply_filter = False






print("Patch size:" + str(patch_size))
log_dir = folder
directory = os.path.dirname(log_dir)
if not os.path.exists(directory):
    os.makedirs(directory)

file = open(log_dir + "holdout" + ".csv", "w+")

X, y = input.read_data(patch_size)
X = X.reshape(len(X), -1)


file.write("\n------------------\nResults for patch size " + str(patch_size) + ":\n")


print('Start training')


X_train, y_train, X_test, y_test = input.read_train_test_data(patch_size)
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)



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


file.write("Size training set: %d\n" % len(X_train))
file.write("Size test set: %d\n" % len(X_test))
file.write("Class distribution:\n")
file.write("Train;Test\n")
dtrain = Counter(y_train)
dtest = Counter(y_test)
for i in range(input.num_classes):
    file.write(str(i) + ";" + str(dtrain[i]) + ";" + str(dtest[i]) + "\n")
file.write("Train acc; Test acc; Test Post acc;Kappa\n")

# clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=None, n_jobs=-1)
# clf = svm.SVC(C=50, gamma=0.01, kernel='rbf')
clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Train accuracy:" + str(accuracy_score(y_train, clf.predict(X_train))))
print("Test accuracy:" + str(acc))


raw, train_acc, test_acc = Decoder.decode_predict(input, patch_size, clf, feature_selector=model)


if apply_filter:
    filt_img, post_test_acc = IndianPines_Postprocessing.apply_modal_filter(input, raw)
else:
    filt_img, post_test_acc = IndianPines_Postprocessing.clean_image(input, raw), test_acc

conf_matrix = IndianPines_Postprocessing.get_conf_matrix(input, filt_img)


fold_oa = conf_matrix.stats_overall['Accuracy']
fold_kappa = conf_matrix.stats_overall['Kappa']

file.write("%.3f;" % train_acc + "%.3f;" % test_acc + "%.3f;" % post_test_acc + "%.3f" % fold_kappa +"\n")

conf_matrix.classification_report.to_csv(log_dir + "classification_report")

save_rgb(log_dir + "outmap.png", filt_img, color_scale=input.color_scale, format='png')

# Clear memory
del X_train, X_test, y_train, y_test


file.close()




