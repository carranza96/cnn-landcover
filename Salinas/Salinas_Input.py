import numpy as np
from sklearn.preprocessing import MinMaxScaler
import spectral.io.envi as envi
from collections import Counter
import scipy.io
from spectral import ColorScale
from imblearn.over_sampling import RandomOverSampler, SMOTE
import math
import tensorflow as tf
from spectral import get_rgb

class Salinas_Input():

    def __init__(self):

        # Load dataset
        trainingset = scipy.io.loadmat("Salinas/Data/Salinas_corrected.mat")['salinas_corrected']
        trainingset_gt = scipy.io.loadmat("Salinas/Data/Salinas_gt.mat")['salinas_gt']



        self.num_classes = 16
        self.class_names = ["Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow",
                            "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
                            "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk",
                            "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained",
                            "Vinyard_vertical_trellis"]

        # Complete ground truth
        self.complete_gt = trainingset_gt


        # Obtain train data
        self.input_data = trainingset
        self.train_data = trainingset_gt
        self.padded_data = self.input_data

        # Dataset variables
        self.height = self.input_data.shape[0]
        self.width = self.input_data.shape[1]
        self.bands = self.input_data.shape[2]
        self.num_pixels = self.height * self.width




        # Store number of pixels training/test
        self.train_pixels = np.count_nonzero(self.train_data)

        # Color scale to display image
        colors =  [ 0,   0,   0, 255,   0,   0,   0, 255,   0,   0,   0, 255, 255, 255,   0,
        0, 255, 255, 255,   0, 255, 176,  48,  96,  46, 139,  87, 160,  32, 240,
        255, 127,  80, 127, 255, 212, 218, 112, 214, 160,  82,  45, 127, 255,   0,
        216, 191, 216, 238,   0,   0]
        class_colors = np.asarray(colors, dtype=int)
        class_colors = class_colors.reshape((int(class_colors.size/3), 3))

        self.color_scale = ColorScale([x for x in range(class_colors.shape[0])], class_colors)




    # Function for obtaining patches
    def Patch(self, patch_size, i, j, pad=False):
        """
        :param i: row index of center of the image patch
        :param j: column index of the center of the image patch
        :return: image patch of size patch_size
        """
        # For every pixel we get 200(number of bands) mini-images (patches) of size 3x3,5x5,... (PATCH_SIZE)
        dist_border = int((patch_size - 1) / 2)  # Distance from center to border of the patch
        if pad:
            return self.padded_data[i - dist_border: i + dist_border + 1, j - dist_border: j + dist_border + 1, :]
        else:
            return self.input_data[i - dist_border: i + dist_border + 1, j - dist_border: j + dist_border + 1, :]



    def read_data(self, patch_size, conv3d=False):

        scaler = MinMaxScaler()
        # Scale: array-like, shape [n_samples, n_features]
        # Flatten input to (145*145,200)
        flat_input = self.input_data.reshape(self.num_pixels, self.bands).astype(float)
        scaled_input = scaler.fit_transform(flat_input)
        # Return to original shape
        self.input_data = scaled_input.reshape(self.height, self.width, self.bands)

        dist_border = int((patch_size - 1) / 2)  # Distance from center to border of the patch

        # Pad data to deal with border pixels
        self.padded_data = np.pad(self.input_data, ((dist_border, dist_border), (dist_border, dist_border), (0, 0)),
                                  'edge')

        # Collect patches of classified pixels
        patches, labels, positions = [], [], []



        for i in range(self.height):
            for j in range(self.width):

                patch = self.Patch(patch_size, i + dist_border, j + dist_border, pad=True)
                label = self.complete_gt[i, j]
                pos = (i, j)

                if label != 0:
                    patches.append(patch)
                    labels.append(label - 1)
                    positions.append(pos)





        # Patches shape: [num_examples, height, width, channels]  (10249,3,3,200) (for 2D Convolution)
        # Final processed dataset: X,y
        X, y = np.asarray(patches, dtype=float), np.asarray(labels, dtype=float)
        positions = np.asarray(positions, dtype=[('i',int),('j',int)])
        return X, y, positions




    # Read patches
    def read_train_test_data(self, patch_size, conv3d=False):
        """
        Function for reading and processing the Indian Pines Dataset
        :return: Processed dataset after collecting classified patches
        """

        # Normalize input using MinMaxScaler (values between 0 and 1)
        scaler = MinMaxScaler()
        # Scale: array-like, shape [n_samples, n_features]
        # Flatten input to (145*145,200)
        flat_input = self.input_data.reshape(self.num_pixels, self.bands).astype(float)
        scaled_input = scaler.fit_transform(flat_input)
        # Return to original shape
        self.input_data = scaled_input.reshape(self.height, self.width, self.bands)

        dist_border = int((patch_size - 1) / 2)  # Distance from center to border of the patch

        # Pad data to deal with border pixels
        self.padded_data = np.pad(self.input_data, ((dist_border, dist_border), (dist_border, dist_border), (0, 0)), 'edge')

        # Collect patches of classified pixels
        train_patches, test_patches = [], []
        train_labels, test_labels = [], []

        for i in range(self.height):
            for j in range(self.width):
                is_train = self.train_data[i, j] != 0
                is_test = self.test_data[i, j] != 0

                if is_train:
                    patch = self.Patch(patch_size, i + dist_border, j + dist_border, pad=True)
                    label = self.train_data[i, j]
                    train_patches.append(patch)
                    train_labels.append(label - 1)

                elif is_test:
                    patch = self.Patch(patch_size, i + dist_border, j + dist_border, pad=True)
                    label = self.test_data[i, j]
                    test_patches.append(patch)
                    test_labels.append(label - 1)


        # Patches shape: [num_examples, height, width, channels]  (10249,3,3,200) (for 2D Convolution)
        # Final processed dataset: X,y
        X_train, X_test = np.asarray(train_patches, dtype=float), np.asarray(test_patches, dtype=float)
        y_train, y_test = np.asarray(train_labels, dtype=int), np.asarray(test_labels, dtype=float)





        # For 3D shape must be 5D Tensor
        # [num_examples, in_depth, in_height, in_width, in_channels(1)]
        if conv3d:
            if rot_oversampling:
                X_train,y_train = self.rotation_oversampling(X_train,y_train)
            X_train, X_test = np.transpose(X_train, axes=(0, 3, 1, 2)), np.transpose(X_test, axes=(0, 3, 1, 2))
            # [num_examples, in_depth, in_height, in_width] Need one more dimension
            X_train, X_test = np.expand_dims(X_train, axis=4), np.expand_dims(X_test, axis=4)

        return X_train, y_train, X_test, y_test



    def oversample_data(self, X, y, patch_size):
        print("Oversampling")
        # ros = SMOTE(random_state=41)
        ros = RandomOverSampler(ratio={10: 400}, random_state=None)
        X, y = ros.fit_sample(X.reshape(len(X), patch_size * patch_size * self.bands), y)
        X = X.reshape(len(X), patch_size, patch_size, self.bands)
        print('Resampled dataset shape {}'.format(Counter(y)))
        return X, y



    def rotation_oversampling(self, X_train, y_train):

        print("Rotating patches")

        # Split to avoid out of mem error
        X_split = np.split(X_train, [2000, 4000])
        y_split = np.split(y_train, [2000, 4000])

        for i in range(len(X_split)):

            tf.reset_default_graph()

            with tf.Graph().as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)

                X = X_split[i]  # Your image or batch of images
                y = y_split[i]
                for degree_angle in [45, 90, 135, 180, 225, 270, 315]:
                #for degree_angle in [45, 90, 180, 270]:
                    radian = degree_angle * math.pi / 180
                    tf_img = tf.contrib.image.rotate(X, radian)
                    rotated_img = sess.run(tf_img)

                    X_train = np.append(X_train, rotated_img, axis=0)
                    y_train = np.append(y_train, y, axis=0)

                sess.close()

        del X_split, y_split


        return X_train, y_train



    def train_test_images(self, train_positions, test_positions):
        img_train, img_test = np.zeros(shape=(self.height, self.width)), np.zeros(shape=(self.height, self.width))

        for (i,j) in train_positions:
            label = self.complete_gt[i, j]
            img_train[i,j] = label

        for (i, j) in test_positions:
            label = self.complete_gt[i, j]
            img_test[i, j] = label


        return get_rgb(img_train, color_scale=self.color_scale), get_rgb(img_test, color_scale=self.color_scale)

