
import spectral.io.envi as envi
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
#import tensorflow as tf
import IndianPines_Input_DFC
from spectral import *

#mat = scipy.io.loadmat("Data/2008_ROSIS_Pavia.mat")
ip_gt = scipy.io.loadmat("Data/Indian_pines_gt.mat")['indian_pines_gt']
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

colores = np.asarray(ts_gt.metadata['class lookup'],dtype=int)
colores = colores.reshape((int(colores.size/3),3))

cs = ColorScale([x for x in range(17)],colores)


rgb_gt_real = get_rgb(ip_gt,color_scale=cs)

rgb_gt = get_rgb(ts_gt.load().read_band(0),color_scale=cs)
rgb = get_rgb(out,color_scale=cs)