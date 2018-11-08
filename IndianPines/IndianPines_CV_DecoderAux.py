import numpy as np
from spectral import get_rgb

def decode(input, train_indices, test_indices, clf):

        patch_size = 1
        X, y = input.read_data(patch_size)
        X = X.reshape(len(X), -1)
        y_pred = clf.predict(X)

        predicted_image = np.zeros(shape=(input.height, input.width))
        correct_pixels_train, correct_pixels_test = [], []

        index = 0

        for i in range(input.height):
            for j in range(input.width):

                label = input.complete_gt[i, j]
                if label != 0:
                    y_ = y_pred[index]
                    predicted_image[i][j] = y_

                    is_train = index in train_indices
                    is_test = index in test_indices

                    if is_train or is_test:
                        index += 1



                if label != 0:
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

        print(len(correct_pixels_train))
        print(len(correct_pixels_test))
        train_acc = np.asarray(correct_pixels_train).mean()*100
        test_acc = np.asarray(correct_pixels_test).mean()*100
        return predicted_image, train_acc, test_acc


def output_image(input, output):
    return get_rgb(output, color_scale=input.color_scale)

