import numpy as np
from sklearn.preprocessing import MinMaxScaler
import spectral.io.envi as envi
from collections import Counter
import scipy.io
from spectral import ColorScale
from imblearn.over_sampling import RandomOverSampler, SMOTE
import math
import tensorflow as tf
import cv2

class Input2018():

    def __init__(self):

        # Load dataset
        self.image = envi.open('IEEEContest2018/Phase2/FullHSIDataset/20170218_UH_CASI_S4_NAD83.hdr',
                                'IEEEContest2018/Phase2/FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix')

        # Change resolution from 1m GSD to 0.5m GSD
        self.resized_image = cv2.resize(self.image.load(), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)



        # Dataset variables
        self.height = self.resized_image.shape[0]
        self.width = self.resized_image.shape[1]
        self.bands = self.resized_image.shape[2]
        self.num_pixels = self.height * self.width

        # Normalize input using MinMaxScaler (values between 0 and 1)
        scaler = MinMaxScaler()
        # Scale: array-like, shape [n_samples, n_features]
        # Flatten input to (145*145,200)
        flat_input = self.resized_image.reshape(self.num_pixels, self.bands).astype(float)
        scaled_input = scaler.fit_transform(flat_input)
        # Return to original shape
        self.resized_image = scaled_input.reshape(self.height, self.width, self.bands)



        self.trainingset_gt = envi.open('IEEEContest2018/Phase2/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.hdr',
                                        'IEEEContest2018/Phase2/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR')

        # Training data variables
        self.train_height = self.trainingset_gt.nrows
        self.train_width = self.trainingset_gt.ncols
        self.train_num_pixels = self.train_height * self.train_width


        x_start_train = int(float(self.trainingset_gt.metadata['map info'][3]) - float(self.image.metadata['map info'][3]))*2
        y_start_train = int(float(self.image.metadata['map info'][4]) - float(self.trainingset_gt.metadata['map info'][4]))*2
        self.trainingset = self.resized_image[x_start_train:x_start_train + self.train_height,
                           y_start_train: y_start_train + self.train_width, :]




        self.num_classes = int(self.trainingset_gt.metadata['classes']) - 1
        self.class_names = self.trainingset_gt.metadata['class names'][1:]


        # Obtain train data
        self.input_data = self.trainingset
        self.train_data = self.trainingset_gt.load().squeeze()
        self.padded_data = self.input_data





        # Store number of pixels training/test
        self.train_pixels = np.count_nonzero(self.train_data)


        # Color scale to display image
        class_colors = np.asarray(self.trainingset_gt.metadata['class lookup'], dtype=int)
        class_colors = class_colors.reshape((int(class_colors.size/3), 3))

        self.color_scale = ColorScale([x for x in range(class_colors.shape[0])], class_colors)



    # Function for obtaining patches
    def Patch(self, input_data, patch_size, i, j):
        """
        :param i: row index of center of the image patch
        :param j: column index of the center of the image patch
        :return: image patch of size patch_size
        """
        # For every pixel we get 200(number of bands) mini-images (patches) of size 3x3,5x5,... (PATCH_SIZE)
        dist_border = int((patch_size - 1) / 2)  # Distance from center to border of the patch
        return input_data[i - dist_border: i + dist_border + 1, j - dist_border: j + dist_border + 1, :]



    # Read patches
    def read_train_data(self, patch_size, pad=True):


        dist_border = int((patch_size - 1) / 2)  # Distance from center to border of the patch

        # Pad data to deal with border pixels
        input_data = self.input_data
        if pad:
            input_data = np.pad(self.input_data, ((dist_border, dist_border), (dist_border, dist_border), (0, 0)), 'edge')

        # Collect patches of classified pixels
        train_patches, train_labels = [], []

        for i in range(self.train_height):
            for j in range(self.train_width):
                patch = self.Patch(input_data, patch_size, i + dist_border, j + dist_border)
                label = self.train_data[i, j]

                if label != 0:  # Ignore patches with unknown landcover type for the central pixel
                    train_patches.append(patch)
                    train_labels.append(label - 1)


        # Patches shape: [num_examples, height, width, channels]  (10249,3,3,200) (for 2D Convolution)
        # Final processed dataset: X,y
        X = np.asarray(train_patches, dtype=float)
        y = np.asarray(train_labels, dtype=int)


        return X, y












    def oversample_data(self, X, y, patch_size):
        print("Oversampling")
        # ros = SMOTE(random_state=41)
        ros = RandomOverSampler(ratio={11: 400}, random_state=37)
        X, y = ros.fit_sample(X.reshape(len(X), patch_size * patch_size * self.bands), y)
        X = X.reshape(len(X), patch_size, patch_size, self.bands)
        print('Resampled dataset shape {}'.format(Counter(y)))
        return X, y



    def rotation_oversampling(self, X_train, y_train):

        print("Rotating patches")

        # Split to avoid out of mem error
        X_split = np.split(X_train, [i * 3000 for i in range(int(len(X_train) / 3000))])
        y_split = np.split(y_train, [i * 3000 for i in range(int(len(X_train) / 3000))])

        for i in range(len(X_split)):

            tf.reset_default_graph()

            with tf.Graph().as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)

                X = X_split[i]  # Your image or batch of images
                y = y_split[i]
                # for degree_angle in [45, 90, 135, 180, 225, 270, 315]:
                for degree_angle in [90, 180, 270]:
                    radian = degree_angle * math.pi / 180
                    tf_img = tf.contrib.image.rotate(X, radian)
                    rotated_img = sess.run(tf_img)

                    X_train = np.append(X_train, rotated_img, axis=0)
                    y_train = np.append(y_train, y, axis=0)

                sess.close()

        del X_split, y_split

        return X_train, y_train
