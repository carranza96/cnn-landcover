import spectral.io.envi as envi
from IEEEContest2018 import Input2018
from spectral import imshow, get_rgb
from scipy import ndimage, stats
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import tensorflow as tf

input = Input2018.Input2018()
img = envi.open('IEEEContest2018/resultados/ps5_4Rot/ps5_4Rotps5.hdr', 'IEEEContest2018/resultados/ps5_4Rot/ps5_4Rotps5.raw').load()

# with tf.Session() as sess:
#     with tf.device('/device:GPU:0'):
# X, y = input.read_train_data(5)

def modal(x):
    return stats.mode(x, axis=None)[0][0]

def mode_filter(img):
    return ndimage.generic_filter(img, modal, size=5)


def output_image(input, output):
    return get_rgb(output, color_scale=input.color_scale)


labelPatches = [patches.Patch(color=input.color_scale.colorTics[x+1]/255., label=input.class_names[x]) for x in range(input.num_classes) ]


view = output_image(input, img)
imshow(view)


print("---------------")
print("Modal filter")
filt_img = img

for n in range(5):
    print("---------------")
    print("Iteration " + str(n))
    filt_img = mode_filter(filt_img)


view = output_image(input, filt_img)
fig = plt.figure(2)
lgd = plt.legend(handles=labelPatches, ncol=1, fontsize='x-small', loc=2, bbox_to_anchor=(1, 1))
imshow(view, fignum=2)



fig.savefig("IEEEContest2018/filt_lgd", bbox_extra_artists=(lgd,), bbox_inches='tight')



envi.save_image("IEEEContest2018/filtro5_5it.hdr", filt_img, dtype='uint8', force=True, interleave='BSQ', ext='raw')

