import random
import numpy as np
from sklearn.utils import shuffle

class DataBuffer():

    def __init__(self, images, labels, batch_size, seed=None):
        """

        :param images: Input data
        :param labels: Input data labels
        :param batch_size: Size of batch
        """

        self.num_examples = images.shape[0]
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

        # Array of indices for shuffling (to avoid having to shuffle all the heavy data)
        self.indices = np.array([i for i in range(self.num_examples)])

        self.epochs_completed = 0
        self.index_in_epoch = 0

        self.seed = seed

    def next_batch(self, shuffle_data=True):
        """
        :param shuffle_data: Boolean for shuffling data at beginning of epoch
        :return: Images and labels batch
        """

        # Shuffle data at the beginning of epoch
        if shuffle_data and self.index_in_epoch == 0:
            np.random.RandomState(self.seed).shuffle(self.indices)
            print(self.indices)
            #self.images,self.labels = get_data_shuffled(self.images,self.labels)


        batch_size, out_range = self.get_out_range_and_batch()


        start_index_batch = self.index_in_epoch
        end_index_batch = start_index_batch + batch_size

        batch_indices = self.indices[start_index_batch:end_index_batch]

        images_batch = self.images[batch_indices]
        labels_batch = self.labels[batch_indices]

        # Reset index if epoch is completed
        if out_range:
            self.index_in_epoch = 0
            self.epochs_completed += 1
        else:
            self.index_in_epoch += batch_size

        return images_batch, labels_batch




    def get_out_range_and_batch(self):
        """
        Checks if remaining input is smaller than default batch size
        :return: Out of range flag, Batch size
        """
        out_range = False
        batch_size = self.batch_size

        if (self.num_examples - self.index_in_epoch) == self.batch_size:
            out_range = True
        elif (self.num_examples - self.index_in_epoch) < self.batch_size:
            batch_size = self.num_examples - self.index_in_epoch
            out_range = True

        return batch_size, out_range






# Static functions

def get_data_shuffled(images,labels):
    """
    :param images: Input data
    :param labels: Input labels data
    :return: Processed shuffled input and labels
    """
    # data = list(zip(images, labels))
    # random.shuffle(data)
    # inputs_processed, labels_processed = zip(*data)
    # return np.asarray(inputs_processed), np.asarray(labels_processed)
    X, y = shuffle(images, labels)
    return X, y
