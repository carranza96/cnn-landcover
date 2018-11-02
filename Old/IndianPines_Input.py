import scipy.io
import numpy as np
from sklearn.preprocessing import MinMaxScaler




"""Routine for reading the Indian Pines dataset provided in .mat file."""



class IndianPines_Input():

    def __init__(self):

        self.input_channels = 200
        self.num_classes = 16
        self.class_names = ["Brocoli_green_weeds_1","Brocoli_green_weeds_2","Fallow","Fallow_rough_plow",
        "Fallow_smooth","Stubble","Celery","Grapes_untrained","Soil_vinyard_develop",
        "Corn_senesced_green_weeds","Lettuce_romaine_4wk","Lettuce_romaine_5wk",
        "Lettuce_romaine_6wk","Lettuce_romaine_7wk","Vinyard_untrained","Vinyard_vertical_trellis"]
        
        # Load dataset
        self.input_data = scipy.io.loadmat("Data/Indian_pines_corrected.mat")['indian_pines_corrected']
        self.target_data = scipy.io.loadmat("Data/Indian_pines_gt.mat")['indian_pines_gt']

        # Dataset variables
        # Input data shape: (145,145,200)
        self.height = self.input_data.shape[0]
        self.width = self.input_data.shape[1]
        self.bands = self.input_data.shape[2]
        self.num_pixels = self.height * self.width



    # Function for obtaining patches
    def Patch(self,patch_size, i, j):
        """
        :param i: row index of center of the image patch
        :param j: column index of the center of the image patch
        :return: image patch of size patch_size
        """
        # For every pixel we get 200(number of bands) mini-images (patches) of size 3x3,5x5,... (PATCH_SIZE)
        dist_border = int((patch_size - 1) / 2)  # Distance from center to border of the patch

        return self.input_data[i - dist_border: i + dist_border + 1, j - dist_border: j + dist_border + 1, :]



    def read_data(self,patch_size,conv3d=False):
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


        # Collect patches of classified pixels
        patches = []
        patches_labels = []

        dist_border = int((patch_size - 1) / 2)  # Distance from center to border of the patch

        for i in range(dist_border, self.height - dist_border):
            for j in range(dist_border, self.width - dist_border):
                patch = self.Patch(patch_size,i, j)
                label = self.target_data[i, j]
                if label != 0:  # Ignore patches with unknown landcover type for the central pixel
                    patches.append(patch)
                    patches_labels.append(label - 1)


        # Patches shape: [num_examples, height, width, channels]  (10249,3,3,200) (for 2D Convolution)
        # Final processed dataset: X,y
        X = np.asarray(patches, dtype=float)
        y = np.asarray(patches_labels, dtype=int)


        # For 3D shape must be 5D Tensor
        # [num_examples, in_depth, in_height, in_width, in_channels(1)]
        if conv3d:
            X = np.transpose(X, axes=(0, 3, 1, 2))
            # [num_examples, in_depth, in_height, in_width] Need one more dimension
            X = np.expand_dims(X, axis=4)

        return X, y
