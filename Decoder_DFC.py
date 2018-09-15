import tensorflow as tf
import CNNModel_2D
import numpy as np
import IndianPines_Input_DFC
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from spectral import get_rgb

# Input data
# input = IndianPines_Input_DFC.IndianPines_Input()
#
# config = {}
# config['patch_size'] = 21
# config['kernel_size'] = 3
# config['conv1_channels'] = 32
# config['conv2_channels'] = 64
# config['fc1_units'] = 1024
# config['batch_size'] = 16
# config['max_epochs'] = 10
# config['train_dropout'] = 0.5
# config['initial_learning_rate'] = 0.01
# config['decaying_lr'] = True
#
# input.read_data(config['patch_size'])



def decode(input,config,model_ckp):

    patch_size = config['patch_size']
    kernel_size = config['kernel_size']
    conv1_channels = config['conv1_channels']
    conv2_channels = config['conv2_channels']
    fc1_units = config['fc1_units']


    #with tf.Graph().as_default():

    # Create placeholders
    images_pl, labels_pl = CNNModel_2D.placeholder_inputs(patch_size, input.bands)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, keep_prob = CNNModel_2D.inference(images_pl, input.bands, patch_size,
                                              kernel_size, conv1_channels, conv2_channels,
                                              fc1_units,input.num_classes)


    softmax = tf.nn.softmax(logits)

    saver = tf.train.Saver()

    sess = tf.Session()

    saver.restore(sess, model_ckp)

    predicted_image = np.zeros(shape=(input.height, input.width))
    correct_pixels_train, correct_pixels_test = [],[]

    dist_border = int((patch_size - 1) / 2)  # Distance from center to border of the patch

    for i in range(input.height):
        for j in range(input.width):

            label = 0
            is_train = input.train_data[i, j] != 0
            is_test = input.test_data[i, j] != 0

            if is_train:
                label = input.train_data[i, j]
            elif is_test:
                label = input.test_data[i, j]

            if label != 0:
                patch = input.Patch(patch_size, i+dist_border, j+dist_border, pad=True)
                patch = np.expand_dims(patch, axis=0)  # Shape [-1,patch_size,patch_size,in_channels]
                predictions = sess.run(softmax, feed_dict={images_pl: patch, keep_prob: 1})
                y_ = np.argmax(predictions) + 1
                predicted_image[i][j] = y_

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

# raw, accuracy = decode(input,config, 'cv21/ps=21,lr_1E-02,lr_d=Y,f=1-model-21.ckpt')
# plt.imshow(raw)
# envi.save_image("ip2.hdr",raw,dtype=int)