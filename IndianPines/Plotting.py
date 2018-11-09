from IndianPines import IndianPines_Input
import matplotlib.pyplot as plt
from spectral import get_rgb,imshow
import numpy as np
from sklearn.utils import shuffle
import math
from pandas_ml import ConfusionMatrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

input = IndianPines_Input.IndianPines_Input()

# plt.figure()
# plt.axis("off")
# img = get_rgb(input.train_data, color_scale=input.color_scale)
#
# imshow(img, fignum=1)

rotation_oversampling = False

X, y = input.read_data(15)
X = X.reshape(len(X), -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


if rotation_oversampling:
    X_train, y_train = input.rotation_oversampling(X_train, y_train)


X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)



for _ in range(5):
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=None, n_jobs=-1)

    clf.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, clf.predict(X_train))
    acc_test = accuracy_score(y_test,clf.predict(X_test))

    print("Train %0.3f" %acc_train)
    print("Test %0.3f" %acc_test)

