import numpy as np
from spectral import get_rgb
from scipy import ndimage, stats
from pandas_ml import ConfusionMatrix




def modal(x):
    return stats.mode(x, axis=None)[0][0]

def mode_filter(img):
    return ndimage.generic_filter(img, modal, size=3)



def accuracy(input, img, train_positions, test_positions):

    correct_pixels_train, correct_pixels_test = [], []

    for (i, j) in train_positions:
        y_ = img[i, j]
        label = input.complete_gt[i, j]
        if label == y_:
            correct_pixels_train.append(1)
        else:
            correct_pixels_train.append(0)

    for (i, j) in test_positions:
        y_ = img[i, j]
        label = input.complete_gt[i, j]
        if label == y_:
            correct_pixels_test.append(1)
        else:
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

            if input.complete_gt[i, j] != 0:
                clean[i, j] = label


    return clean


def get_conf_matrix(input, img, test_positions):

    test_labels, test_predictions = [], []

    for (i, j) in test_positions:
        y_ = img[i, j]
        label = input.complete_gt[i, j]

        test_labels.append(label)
        test_predictions.append(y_)

    conf_matrix = ConfusionMatrix(test_labels, test_predictions)

    return conf_matrix




def apply_modal_filter(input, img, train_positions, test_positions):

    filt_img = img

    print("----------\nBefore filter")
    train_acc, test_acc = accuracy(input, filt_img, train_positions, test_positions)
    print("Training accuracy: %.2f" % train_acc)
    print("Test accuracy: %.2f" % test_acc)

    for n in range(5):
        print("---------------")
        print("Iteration " + str(n))
        filt_img = mode_filter(filt_img)

        train_acc, test_acc = accuracy(input, filt_img, train_positions, test_positions)
        print("Training accuracy: %.2f" %train_acc)
        print("Test accuracy: %.2f" %test_acc)

    # clean_img = clean_image(input, filt_img)
    clean_img = filt_img
    return clean_img, test_acc


