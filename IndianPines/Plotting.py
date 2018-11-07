from IndianPines import IndianPines_Input
import matplotlib.pyplot as plt
from spectral import get_rgb,imshow
import numpy as np
from sklearn.utils import shuffle
import math
from pandas_ml import ConfusionMatrix
import pandas as pd

input = IndianPines_Input.IndianPines_Input()

# plt.figure()
# plt.axis("off")
# img = get_rgb(input.train_data, color_scale=input.color_scale)
#
# imshow(img, fignum=1)

X, y = input.read_data(5)

indices = {c: np.where(y==c)[0] for c in range(input.num_classes)}

groups = np.zeros(shape=y.shape)

for c in indices.keys():
    np.put(groups, indices[c], np.asarray([int(math.floor(i / 50)) + 1 for i in range(len(indices[c]))]))



y1, y1pred= [1,2,3,4,1,1],[1,2,2,2,3,4]
y2, y2pred = [2,3,4,1,2,2],[4,4,2,1,3,3]

cm1 = ConfusionMatrix(y1, y1pred)
cm2 = ConfusionMatrix(y2, y2pred)

cs1 = cm1.classification_report
cs2 = cm2.classification_report

df_concat = pd.concat((cs1,cs2)).astype('float')

by_row_index = df_concat.groupby(df_concat.index)
# df_means = by_row_index.meancs()