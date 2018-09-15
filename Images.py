
import spectral.io.envi as envi
import scipy.io
import numpy as np
import IndianPines_Input_DFC
from spectral import imshow,get_rgb


#mat = scipy.io.loadmat("Data/2008_ROSIS_Pavia.mat")

input = IndianPines_Input_DFC.IndianPines_Input()


# view = imshow(input.train_data, color_scale=input.color_scale)
# view = imshow(input.test_data, color_scale=input.color_scale)


view = imshow(input.complete_gt, color_scale=input.color_scale, classes=input.train_data)
view.set_display_mode('overlay')
view.class_alpha = 0.4
# view = imshow(input.test_data, color_scale=input.color_scale)









# plt.imshow(img[:,:,20], interpolation='nearest')
# plt.show()



# plt.figure()
# plt.imshow(ts_gt.load().squeeze())
# plt.figure()
# plt.imshow(ts.load().read_band(10))

#
# outputmap = envi.open('ip2.hdr', 'ip2.img')
#
#
#
#
#
#
rgb_gt2 = get_rgb(ts_gt.load().read_band(0),color_scale=cs)
# rgb = get_rgb(outputmap,color_scale=cs)
#
#
# count = 0
# total = np.count_nonzero(gt_converted)
# test = 0
# print(total)
# for i in range(145):
#     for j in range(145):
#         gtval = gt_converted[i][j]
#         out = outputmap.read_pixel(i,j)
#         is_test = gtval!=0 and ts_gt.read_pixel(i,j)==0
#         if is_test:
#             if gtval == out:
#                 count+=1
#             test+=1
#
# print(test)
# acc = count/test
# print("Accuracy: ",acc*100)

