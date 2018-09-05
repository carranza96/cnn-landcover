


import scipy.io
import numpy as np
from matplotlib import pyplot as plt
#import tensorflow as tf

mat = scipy.io.loadmat("Data/2008_ROSIS_Pavia.mat")

img = mat['image']

plt.imshow(img[:,:,20], interpolation='nearest')
plt.show()


