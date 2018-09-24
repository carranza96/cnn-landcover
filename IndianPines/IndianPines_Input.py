import numpy as np
from sklearn.preprocessing import MinMaxScaler
import spectral.io.envi as envi
from collections import Counter
import scipy.io
from spectral import ColorScale
from imblearn.over_sampling import RandomOverSampler,SMOTE
import math
import tensorflow as tf

class IndianPines_Input():

    def __init__(self):

        # Load dataset
        trainingset = envi.open('IndianPines/Data/indianpines_ds_raw.hdr', 'IndianPines/Data/indianpines_ds_raw.raw')
        trainingset_gt = envi.open('IndianPines/Data/indianpines_ts_raw_classes.hdr', 'IndianPines/Data/indianpines_ts_raw_classes.raw')

        # Dataset variables
        # Input data shape: (145,145,200)
        self.height = trainingset.nrows
        self.width = trainingset.ncols
        self.bands = trainingset.nbands
        self.num_pixels = self.height * self.width

        self.num_classes = int(trainingset_gt.metadata['classes']) - 1
        self.class_names = trainingset_gt.metadata['class names'][1:]

        # Complete ground truth
        self.complete_gt = self.convert_gt(scipy.io.loadmat("IndianPines/Data/Indian_pines_gt.mat")['indian_pines_gt'])

        # Obtain train data
        self.input_data = trainingset.load()
        self.train_data = trainingset_gt.load().squeeze()
        self.padded_data = self.input_data



        # Obtain test data by comparing training set to complete ground truth
        self.test_data = self.get_test_data()


        # Store number of pixels training/test
        self.train_pixels = np.count_nonzero(self.train_data)
        self.test_pixels = np.count_nonzero(self.test_data)

        # Color scale to display image
        class_colors = np.asarray(trainingset_gt.metadata['class lookup'], dtype=int)
        class_colors = class_colors.reshape((int(class_colors.size/3), 3))

        self.color_scale = ColorScale([x for x in range(class_colors.shape[0])], class_colors)





    # Convert complete ground truth data to have same class code as in training gt image
    def convert_gt(self, gt):
        gt_converter = {0: 0, 1: 10, 2: 1, 3: 2, 4: 11, 5: 3, 6: 4, 7: 12, 8: 5, 9: 13, 10: 6, 11: 7, 12: 8, 13: 14,
                        14: 9, 15: 15, 16: 16}

        gt_converted = np.zeros(shape=(self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                gt_converted[i][j] = gt_converter[gt[i][j]]

        return gt_converted


    # Create test data ground truth image by comparing the complete gt to the training set gt
    def get_test_data(self):
        test_data = np.zeros(shape=(self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                real_class = self.complete_gt[i][j]
                is_training = self.train_data[i][j] != 0
                if real_class != 0 and not is_training:
                    test_data[i][j] = self.complete_gt[i][j]

        return test_data



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



    # Read patches
    def read_data(self, patch_size, conv3d=False):
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
            X_train, X_test = np.transpose(X_train, axes=(0, 3, 1, 2)), np.transpose(X_test, axes=(0, 3, 1, 2))
            # [num_examples, in_depth, in_height, in_width] Need one more dimension
            X_train, X_test = np.expand_dims(X_train, axis=4), np.expand_dims(X_test, axis=4)

        return X_train, y_train, X_test, y_test



    def oversample_data(self, X, y, patch_size):
        print("Oversampling")
        # ros = SMOTE(random_state=41)
        ros = RandomOverSampler(ratio={11:300} ,random_state=41)
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
                # for degree_angle in [90, 180, 270]:
                    radian = degree_angle * math.pi / 180
                    tf_img = tf.contrib.image.rotate(X, radian)
                    rotated_img = sess.run(tf_img)

                    X_train = np.append(X_train, rotated_img, axis=0)
                    y_train = np.append(y_train, y, axis=0)

                sess.close()

        del X_split, y_split

        return X_train, y_train
