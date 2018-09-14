import tensorflow as tf
import CNNModel_2D
import numpy as np
import IndianPines_Input_DFC
import matplotlib.pyplot as plt
import spectral.io.envi as envi

# Input data
input = IndianPines_Input_DFC.IndianPines_Input()

config = {}
config['patch_size'] = 21
config['kernel_size'] = 3
config['conv1_channels'] = 32
config['conv2_channels'] = 64
config['fc1_units'] = 1024
config['batch_size'] = 16
config['max_epochs'] = 10
config['train_dropout'] = 0.5
config['initial_learning_rate'] = 0.01
config['decaying_lr'] = True

input.read_data(config['patch_size'])



def decode(input,config,model_ckp):

    patch_size = config['patch_size']
    kernel_size = config['kernel_size']
    conv1_channels = config['conv1_channels']
    conv2_channels = config['conv2_channels']
    fc1_units = config['fc1_units']


    #with tf.Graph().as_default():

    # Create placeholders
    images_pl, labels_pl = CNNModel_2D.placeholder_inputs(patch_size, input.input_channels)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, keep_prob = CNNModel_2D.inference(images_pl, input.input_channels,patch_size,
                                              kernel_size, conv1_channels, conv2_channels,
                                              fc1_units,input.num_classes)


    softmax = tf.nn.softmax(logits)

    saver = tf.train.Saver()

    sess = tf.Session()

    saver.restore(sess, model_ckp)

    predicted_image = np.zeros(shape=(input.height, input.width))
    correct_pixels = []

    dist_border = int((patch_size - 1) / 2)  # Distance from center to border of the patch

    for i in range(input.height):
        for j in range(input.width):
            label = input.target_data[i, j]

            patch = input.Patch(patch_size, i+dist_border, j+dist_border, pad=True)
            patch = np.expand_dims(patch, axis=0)  # Shape [-1,patch_size,patch_size,in_channels]
            predictions = sess.run(softmax, feed_dict={images_pl: patch, keep_prob: 1})
            y_ = np.argmax(predictions) + 1
            predicted_image[i][j] = y_

            if label == y_:
                correct_pixels.append(1)
            else:
                correct_pixels.append(0)

    accuracy = np.asarray(correct_pixels).mean()*100

    return predicted_image, accuracy



# raw, accuracy = decode(input,config, 'cv21/ps=21,lr_1E-02,lr_d=Y,f=1-model-21.ckpt')
# plt.imshow(raw)
# envi.save_image("ip2.hdr",raw,dtype=int)