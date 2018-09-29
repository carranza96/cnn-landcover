from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math


def placeholder_inputs(patch_size, in_depth):
    """
    :param patch_size: Size of patch (small image to feed the network)
    :return: Images and labels placeholders
    """
    in_channels = 1  # Only one cube with #in_depth bands, would be more if we had multitemporal information

    x = tf.placeholder(tf.float32, shape=[None, in_depth, patch_size, patch_size, in_channels])
    y_ = tf.placeholder(tf.int64, shape=[None])
    phase_train = tf.placeholder(tf.bool, name='phase') # Placerholder for batch_norm, training (True) or not training (False)

    return x, y_, phase_train



def inference(images, in_depth, in_channels, patch_size, spectral_kernel_size,spatial_kernel_size, conv1_channels,conv2_channels,fc1_units, number_of_classes, phase_train):
    """
    :param images:  Images placeholder
    :param in_channels: Number of channels of image
    :param patch_size: Size of patch (small image to feed the network)
    :param spectral_kernel_size: Depth of convolutional kernel (number of bands)
    :param spatial_kernel_size: Size of convolutional kernel (height and width)
    :param conv1_channels: Size of the first hidden layer.
    :param conv2_channels: Size of the second hidden layer.
    :param fc1_units: Size of the fully connected layer.
    :param number_of_classes:
    :param phase_train:
    :return:
        Output tensor with the computed logits.
        Scalar placeholder for the probability of dropout.
    """

    # Input Layer
    # Reshape X to 5-D tensor:  [batch_size, in_depth(# bands),in_height,in_width, in_channels(1)]
    with tf.name_scope('reshape'):
        x_image = tf.reshape(images, [-1, in_depth, patch_size, patch_size, in_channels])



    with tf.name_scope('bn1'):
        h_bn1 = batch_norm(x_image, phase_train)


    # Convolutional Layer #1
    # Maps the 200 patches to conv1_channels feature maps.
    # Padding is "same" to preserve width and height
    # Input Tensor Shape: [batch_size, in_depth,patch_size, patch_size, in_channels(1)]
    # Output Tensor Shape: [batch_size, in_depth, patch_size, patch_size, conv1_channels]
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([spectral_kernel_size, spatial_kernel_size, spatial_kernel_size, in_channels, conv1_channels])
        b_conv1 = bias_variable([conv1_channels])
        h_conv1 = tf.nn.relu(conv3d(h_bn1, W_conv1) + b_conv1)



    with tf.name_scope('bn2'):
        h_bn2 = batch_norm(h_conv1, phase_train)

    # Pooling layer #1
    # Downsamples by 2X.
    # Input Tensor Shape: [batch_size, in_depth, patch_size, patch_size, conv1_channels]
    # Output Tensor Shape: [batch_size, in_depth/2, patch_size/2 , patch_size/2 , conv1_channels]
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_3d_2x2(h_bn2)

    # Convolutional Layer #2
    # Computes conv2_channels features using a kernel_size filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, in_depth/2, patch_size/2 , patch_size/2 , conv1_channels]
    # Output Tensor Shape: [batch_size, in_depth/2, patch_size/2 , patch_size/2 , conv2_channels]
    with tf.name_scope('conv2'):
      W_conv2 = weight_variable([spectral_kernel_size,spatial_kernel_size, spatial_kernel_size, conv1_channels, conv2_channels])
      b_conv2 = bias_variable([conv2_channels])
      h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)


    with tf.name_scope('bn3'):
        h_bn3 = batch_norm(h_conv2, phase_train)

    # Pooling layer #2
    # Input Tensor Shape: [batch_size, in_depth/2, patch_size/2 , patch_size/2 , conv2_channels]
    # Output Tensor Shape: [batch_size, in_depth/4, patch_size/4 , patch_size/4 , conv2_channels]
    with tf.name_scope('pool2'):
      h_pool2 = max_pool_3d_2x2(h_bn3)


    # Fully connected layer 1 (Dense layer)
    # After 2 round of downsampling, our in_depth*patch_size*patch_size image
    # is down to (in_depth/4)*(patch_size/4)*(patch_size/4)*conv2_channels feature maps -- maps this to fc1_units features.
    # Input Tensor Shape: [batch_size, (in_depth/4)*(patch_size/4)*(patch_size/4)*conv2_channels]
    # Output Tensor Shape: [batch_size, fc1_units]
    with tf.name_scope('fc1'):
      # dim = np.prod(h_pool2.get_shape().as_list()[1:])
      # prev_layer_flat = tf.reshape(h_pool2, [-1, dim])

      size_after_pools = math.ceil(patch_size/4) # Padding = SAME
      depth_after_pools = math.ceil(in_depth/4)
      W_fc1 = weight_variable([depth_after_pools*size_after_pools * size_after_pools * conv2_channels, fc1_units])
      b_fc1 = bias_variable([fc1_units])

      h_pool2_flat = tf.reshape(h_pool2, [-1, depth_after_pools * size_after_pools * size_after_pools * conv2_channels])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
      keep_prob = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Logits Layer
    # Map the fc1_units features to the number of classes of the problem
    # Input Tensor Shape: [batch_size, fc1_units]
    # Output Tensor Shape: [batch_size, num_clasess]
    with tf.name_scope('softmax_linear'):
      W_fc2 = weight_variable([fc1_units, number_of_classes])
      b_fc2 = bias_variable([number_of_classes])

      logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return logits, keep_prob




def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv3d(x, W):
  """conv3d returns a 3d convolution layer with full stride."""
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_3d_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME')


def batch_norm(x, phase):
    return tf.layers.batch_normalization(x, center=True, scale=True, training=phase)








def loss(logits, labels):
  """
  :param logits: Logits from inference(). Shape [batch_size,num_classes]
  :param labels: Labels from inputs(). 1-D tensor of shape [batch_size]

  :return: Loss tensor of type float.
  """

  # Calculate the average cross entropy loss across the batch.
  # labels = tf.one_hot(indices=tf.cast(labels, tf.int64), depth=number_of_classes)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
  return cross_entropy_mean


## A lo mejor le tengo que meter global_step como parámetro para rate decay y moving average
def training(loss, learning_rate, global_step):
    """
    Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    sess.run() call to cause the model to train.

    :param loss: Loss tensor, from loss().
    :param learning_rate: The learning rate to use for gradient descent.

    :return: The Op for training.
    """

    with tf.name_scope("train"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            # Create the gradient descent optimizer with the given learning rate.
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            # Use the optimizer to apply the gradients that minimize the loss
            # (and also increment the global step counter) as a single training step.
            train_op = optimizer.minimize(loss, global_step=global_step)

            tf.summary.scalar('learning_rate', learning_rate, collections=['train'])

    return train_op





def evaluation(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label.
    :param logits:  Logits from inference(). Shape [batch_size,num_classes]
    :param labels: Labels from inputs(). 1-D tensor of shape [batch_size]
    :return: Accuracy of the model
    """

    predictions = tf.argmax(input=logits, axis=1)
    correct_predictions = tf.cast(tf.equal(predictions, labels), tf.float32)
    accuracy = tf.reduce_mean(correct_predictions)

    return predictions, correct_predictions, accuracy








