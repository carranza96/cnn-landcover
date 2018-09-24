import spectral.io.envi as envi
import IndianPines_Input_DFC
import numpy as np
from spectral import imshow, get_rgb
from scipy import ndimage, stats
import matplotlib.patches as patches
import matplotlib.pyplot as plt

input = IndianPines_Input_DFC.IndianPines_Input()

# img = envi.open('mejor_resultado/ps5.hdr', 'mejor_resultado/ps5.raw')
img = envi.open('mejor_resultado/Con batch norm/ip_filtro3_5it.hdr', 'mejor_resultado/Con batch norm/ip_filtro3_5it.raw')

def modal(x):
    return stats.mode(x, axis=None)[0][0]

def mode_filter(img):
    return ndimage.generic_filter(img, modal, size=3)



def accuracy(input, img):

    correct_pixels_train, correct_pixels_test = [], []

    for i in range(input.height):
        for j in range(input.width):

            y_ = img[i, j]

            label = 0
            is_train = input.train_data[i, j] != 0
            is_test = input.test_data[i, j] != 0

            if is_train:
                label = input.train_data[i, j]
            elif is_test:
                label = input.test_data[i, j]

            if label == y_:
                if is_train:
                    correct_pixels_train.append(1)
                elif is_test:
                    correct_pixels_test.append(1)
            else:
                if is_train:
                    correct_pixels_train.append(0)
                elif is_test:
                    correct_pixels_test.append(0)


    train_acc = np.asarray(correct_pixels_train).mean() * 100
    test_acc = np.asarray(correct_pixels_test).mean() * 100
    return train_acc, test_acc


def output_image(input, output):
    return get_rgb(output, color_scale=input.color_scale)


def clean_image(input, img):
    clean = np.zeros(shape=(input.height, input.width))

    for i in range(input.height):
        for j in range(input.width):

            label = img[i, j]

            is_train = input.train_data[i, j] != 0
            is_test = input.test_data[i, j] != 0

            if is_train or is_test:
                clean[i, j] = label


    return clean



train_acc, test_acc = accuracy(input,img)
view = output_image(input, img)
# imshow(view)
clean_img = clean_image(input, img)
view = output_image(input, clean_img)
# imshow(view)
labelPatches = [patches.Patch(color=input.class_colors[x]/255., label=input.class_names[x]) for x in range(input.num_classes) ]
plt.figure()
plt.legend(handles=labelPatches, ncol=2, fontsize='medium', loc='right', bbox_to_anchor=(0.5, -0.05));
plt.imshow(img.read_band(0))

print("Training accuracy: %.2f" %train_acc)
print("Test accuracy: %.2f" %test_acc)
#
# print("---------------")
# print("Modal filter")
# filt_img = img.load()
#
# for n in range(5):
#     print("---------------")
#     print("Iteration " + str(n))
#     filt_img = mode_filter(filt_img)
#
#     train_acc, test_acc = accuracy(input, filt_img)
#     print("Training accuracy: %.2f" %train_acc)
#     print("Test accuracy: %.2f" %test_acc)
#
# view = output_image(input, filt_img)
# imshow(view)
#
# clean_img = clean_image(input, filt_img)
# view = output_image(input, clean_img)
# print(input.class_names)
# imshow(view)


# envi.save_image("ip_filtro3_5it.hdr", filt_img, dtype='uint8', force=True, interleave='BSQ', ext='raw')

