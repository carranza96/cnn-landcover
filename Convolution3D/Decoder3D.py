import tensorflow as tf
import CNNModel_3D
import numpy as np
from spectral import get_rgb



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


def decode(input, config,train_positions, test_positions, model_ckp):
    patch_size = config['patch_size']
    in_depth = config['in_depth']
    in_channels= config['in_channels']
    spectral_kernel_size = config['spectral_kernel_size']
    spatial_kernel_size = config['spatial_kernel_size']
    conv1_channels = config['conv1_channels']
    conv2_channels = config['conv2_channels']
    fc1_units = config['fc1_units']

    tf.reset_default_graph()

    with tf.Graph().as_default():

        # Create placeholders
        images_pl, labels_pl, phase_train = CNNModel_3D.placeholder_inputs(patch_size, input.bands)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, keep_prob = CNNModel_3D.inference(images_pl, in_depth, in_channels, patch_size,
                                                  spectral_kernel_size, spatial_kernel_size, conv1_channels, conv2_channels,
                                                  fc1_units, input.num_classes, phase_train)

        softmax = tf.nn.softmax(logits)

        saver = tf.train.Saver()

        sess = tf.Session()

        saver.restore(sess, model_ckp)

        predicted_image = np.zeros(shape=(input.height, input.width))

        dist_border = int((patch_size - 1) / 2)  # Distance from center to border of the patch

        for i in range(input.height):
            col_patches = []

            for j in range(input.width):
                patch = input.Patch(patch_size, i + dist_border, j + dist_border, pad=True)
                col_patches.append(patch)

            col_patches = np.asarray(col_patches)
            col_patches = np.transpose(col_patches, axes=(0, 3, 1, 2))
            col_patches = np.expand_dims(col_patches, axis=4)

            predictions = sess.run(softmax, feed_dict={images_pl: col_patches, keep_prob: 1, phase_train: False})
            y_ = [np.argmax(pred) + 1 for pred in predictions]
            predicted_image[i] = y_

        sess.close()

        train_acc, test_acc = accuracy(input, predicted_image, train_positions, test_positions)

        return predicted_image, train_acc, test_acc



def output_image(input, output):
    return get_rgb(output, color_scale=input.color_scale)

