import numpy as np
from spectral import get_rgb

def decode(input, patch_size, train_indices, test_indices, clf):

        # X, y = input.read_data(patch_size)
        # X = X.reshape(len(X), -1)
        # y_pred = clf.predict(X)

        patches = []
        dist_border = int((patch_size - 1) / 2)  # Distance from center to border of the patch


        for i in range(input.height):
            for j in range(input.width):
                patch = input.Patch(patch_size, i + dist_border, j + dist_border, pad=True)
                patch = patch.reshape(1, -1)
                patches.append(patch)

        patches = np.asarray(patches).reshape(len(patches), -1)
        y_pred = clf.predict(patches)

        predicted_image = np.zeros(shape=(input.height, input.width))
        correct_pixels_train, correct_pixels_test = [], []

        index = 0

        for i in range(input.height):
            for j in range(input.width):

                y_ = y_pred[i*input.height + j] + 1

                predicted_image[i][j] = y_

                label = input.complete_gt[i, j]


                if label != 0:
                    is_train = index in train_indices
                    is_test = index in test_indices

                    if is_train or is_test:
                        index += 1


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


        train_acc = np.asarray(correct_pixels_train).mean()*100
        test_acc = np.asarray(correct_pixels_test).mean()*100
        return predicted_image, train_acc, test_acc


def output_image(input, output):
    return get_rgb(output, color_scale=input.color_scale)

