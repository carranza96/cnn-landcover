from PIL import Image
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



class Flevoland_Input():

    def __init__(self):

        # Load dataset
        trainingset = envi.open('Flevoland/Data/Flevoland.hdr',
                                'Flevoland/Data/Flevoland.raw')

        # trainingset_gt = np.array(Image.open('Flevoland/Data/TrainingSet/cm3253_ground_truth_training.tiff'))
        trainingset_gt = envi.open('Flevoland/Data/Flevoland_gt_corrected.hdr','Flevoland/Data/Flevoland_gt_corrected.raw')

        # Dataset variables
        # Input data shape: (145,145,200)
        self.height = trainingset.nrows
        self.width = trainingset.ncols
        self.bands = trainingset.nbands
        self.num_pixels = self.height * self.width

        self.num_classes = int(trainingset.metadata['classes']) - 1
        self.class_names = trainingset.metadata['class names'][1:]

        # Complete ground truth
        # self.complete_gt = self.convert_gt(scipy.io.loadmat("IndianPines/Data/Indian_pines_gt.mat")['indian_pines_gt'])

        # Obtain train data
        self.input_data = trainingset.load()
        self.train_data = trainingset_gt.load().squeeze()
        self.padded_data = self.input_data

        self.complete_gt = self.train_data

        # Obtain test data by comparing training set to complete ground truth
        # self.test_data = self.get_test_data()


        # Store number of pixels training/test
        self.train_pixels = np.count_nonzero(self.train_data)
        # self.test_pixels = np.count_nonzero(self.test_data)

        # Color scale to display image
        class_colors = np.asarray(trainingset.metadata['class lookup'], dtype=int)
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
        train_patches, train_labels,positions = [], [], []

        for i in range(self.height):
            for j in range(self.width):
                patch = self.Patch(patch_size, i + dist_border, j + dist_border, pad=True)
                label = self.train_data[i, j]
                pos = (i, j)

                if label != 0:  # Ignore patches with unknown landcover type for the central pixel
                    train_patches.append(patch)
                    train_labels.append(label - 1)
                    positions.append(pos)



        # Patches shape: [num_examples, height, width, channels]  (10249,3,3,200) (for 2D Convolution)
        # Final processed dataset: X,y
        X_train = np.asarray(train_patches, dtype=float)
        y_train = np.asarray(train_labels, dtype=int)
        positions = np.asarray(positions, dtype=[('i',int),('j',int)])


        # For 3D shape must be 5D Tensor
        # [num_examples, in_depth, in_height, in_width, in_channels(1)]
        if conv3d:
            X_train = np.transpose(X_train, axes=(0, 3, 1, 2))
            # [num_examples, in_depth, in_height, in_width] Need one more dimension
            X_train= np.expand_dims(X_train, axis=4)

        return X_train, y_train, positions



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
        X_split = np.split(X_train, [i*2000 for i in range(int(len(X_train)/2000))])
        y_split = np.split(y_train, [i*2000 for i in range(int(len(X_train)/2000))])

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




    def train_test_images(self, train_positions, test_positions):
        img_train, img_test = np.zeros(shape=(self.height, self.width)), np.zeros(shape=(self.height, self.width))

        for (i,j) in train_positions:
            label = self.complete_gt[i, j]
            img_train[i,j] = label

        for (i, j) in test_positions:
            label = self.complete_gt[i, j]
            img_test[i, j] = label


        return get_rgb(img_train, color_scale=self.color_scale), get_rgb(img_test, color_scale=self.color_scale)