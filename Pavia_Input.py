
import spectral.io.envi as envi
import scipy.io
import numpy as np
import IndianPines_Input_DFC
from spectral import *

#mat = scipy.io.loadmat("Data/2008_ROSIS_Pavia.mat")
ip_gt = scipy.io.loadmat("Data/Indian_pines_gt.mat")['indian_pines_gt']
colors_gt = [[0,0,0],
 [255, 127,  80],
 [255,   0,   0],
 [0, 255,   0],
 [127, 255, 212],
 [0,   0, 255],
 [255, 255,   0],
 [218, 112, 214],
 [0, 255, 255],
 [160,  82,  45],
 [255,   0, 255],
 [176,  48,  96],
 [46, 139,  87],
 [127, 255,   0],
 [160,  32, 240],
 [216, 191, 216],
 [238,   0,   0]]
cs_gt = ColorScale([x for x in range(17)],np.asanyarray(colors_gt))

rgb_gt = get_rgb(ip_gt,color_scale=cs_gt)


converter = {0:0,
             1:10,
             2:1,
             3:2,
             4:11,
             5:3,
             6:4,
             7:12,
             8:5,
             9:13,
             10:6,
             11:7,
             12:8,
             13:14,
             14:9,
             15:15,
             16:16}

gt_converted = np.zeros(shape=(145,145))
for i in range(145):
    for j in range(145):
        gt_converted[i][j] = converter[ip_gt[i][j]]



# plt.imshow(img[:,:,20], interpolation='nearest')
# plt.show()


# ts = envi.open('Data/IP_DataSet/indianpines_ds_raw.hdr', 'Data/IP_DataSet/indianpines_ds_raw.raw')
ts_gt = envi.open('Data/IP_TraingSet/indianpines_ts_raw_classes.hdr', 'Data/IP_TraingSet/indianpines_ts_raw_classes.raw')
# plt.figure()
# plt.imshow(ts_gt.load().squeeze())
# plt.figure()
# plt.imshow(ts.load().read_band(10))


outputmap = envi.open('ip.hdr', 'ip.raw')


colores = np.asarray(ts_gt.metadata['class lookup'],dtype=int)
colores = colores.reshape((int(colores.size/3),3))

cs = ColorScale([x for x in range(17)],colores)




rgb_gt2 = get_rgb(ts_gt.load().read_band(0),color_scale=cs)
rgb = get_rgb(outputmap,color_scale=cs)


count = 0
total = np.count_nonzero(gt_converted)
test = 0
print(total)
for i in range(145):
    for j in range(145):
        gtval = gt_converted[i][j]
        out = outputmap.read_pixel(i,j)
        is_test = gtval!=0 and ts_gt.read_pixel(i,j)==0
        if is_test:
            if gtval == out:
                count+=1
            test+=1

print(test)
acc = count/test
print("Accuracy: ",acc*100)

