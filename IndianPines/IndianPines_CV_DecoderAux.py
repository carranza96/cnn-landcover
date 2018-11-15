import numpy as np
from spectral import get_rgb

def decode(input, patch_size, clf, feature_selector=None):


        patches = []
        dist_border = int((patch_size - 1) / 2)  # Distance from center to border of the patch


        for i in range(input.height):
            for j in range(input.width):
                patch = input.Patch(patch_size, i + dist_border, j + dist_border, pad=True)
                patch = patch.reshape(1, -1)
                patches.append(patch)

        patches = np.asarray(patches).reshape(len(patches), -1)
        if feature_selector:
            patches = feature_selector.transform(patches)

        y_pred = clf.predict(patches)


        predicted_image = np.zeros(shape=(input.height, input.width))


        for i in range(input.height):
            for j in range(input.width):

                y_ = y_pred[i*input.width + j] + 1

                predicted_image[i][j] = y_


        return predicted_image


def output_image(input, output):
    return get_rgb(output, color_scale=input.color_scale)

