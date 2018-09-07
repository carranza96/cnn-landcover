
import spectral.io.envi as envi
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
#import tensorflow as tf
import IndianPines_Input_DFC
from spectral  import *

#mat = scipy.io.loadmat("Data/2008_ROSIS_Pavia.mat")

#img = mat['image']

# plt.imshow(img[:,:,20], interpolation='nearest')
# plt.show()


# ts = envi.open('Data/IP_DataSet/indianpines_ds_raw.hdr', 'Data/IP_DataSet/indianpines_ds_raw.raw')
ts_gt = envi.open('Data/IP_TraingSet/indianpines_ts_raw_classes.hdr', 'Data/IP_TraingSet/indianpines_ts_raw_classes.raw')
# plt.figure()
# plt.imshow(ts_gt.load().squeeze())
# plt.figure()
# plt.imshow(ts.load().read_band(10))


out = envi.open('ip.hdr', 'ip.raw')

colors = np.asarray(ts_gt.metadata['class lookup'],dtype=int)
colors = colors.reshape((int(colors.size/3),3))