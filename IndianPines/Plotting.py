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

X, y = input.read_data(5)
X = X.reshape(len(X),5*5*220)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test,y_pred)

print(acc)