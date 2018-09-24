import time

import tensorflow as tf

import CNNModel_2D
from DataBuffer import *
import math
#from sklearn.metrics import confusion_matrix
from pandas_ml import ConfusionMatrix


def train_model(X_train,y_train,X_test,y_test,config):
    """
    Trains the model and keeps track of how the network predictions improve
    :param X_train: Training data
    :param y_train: Training data labels
    :param X_test: Test data
    :param y_test: Test data labels
    :param config: Parameter configuration for the network
    :return:
    """
    patch_size = config['patch_size']
    in_channels = config['in_channels']
    num_classes = config['num_classes']
    kernel_size = config['kernel_size']
    conv1_channels = config['conv1_channels']
    conv2_channels = config['conv2_channels']
    fc1_units = config['fc1_units']
    batch_size =  config['batch_size']
    max_epochs = config['max_epochs']
    train_dropout = config['train_dropout']
    initial_learning_rate = config['initial_learning_rate']
    decaying_lr = config['decaying_lr']
    log_dir = config['log_dir']


    tf.reset_default_graph()

    """Train Indian Pines for a number of steps."""
    with tf.Graph().as_default():

        # Size of input
        input_size = len(y_train)
        num_batches_per_epoch = int(math.ceil(input_size / batch_size))
        test_size = len(y_test)

        # Create placeholders
        images_pl, labels_pl, phase_train = CNNModel_2D.placeholder_inputs(patch_size,in_channels)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, keep_prob = CNNModel_2D.inference(images_pl, in_channels, patch_size, kernel_size, conv1_channels,
                                                  conv2_channels, fc1_units, num_classes, phase_train)

        # Calculate loss.
        loss = CNNModel_2D.loss(logits, labels_pl)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Define learning rate
        # Decay once per epoch , using an exponential schedule starting at initial_learning_rate
        if decaying_lr:
            learning_rate = tf.train.exponential_decay(
                initial_learning_rate,  # Base learning rate.
                global_step,  # Current index into the dataset.
                num_batches_per_epoch,  # Decay step.
                0.96,  # Decay rate.
                staircase=True)
        else:
            learning_rate = initial_learning_rate

        train_step = CNNModel_2D.training(loss, learning_rate, global_step)

        # Add the Op to compare the logits to the labels during evaluation.
        predictions, correct_predictions, accuracy = CNNModel_2D.evaluation(logits, labels_pl)


        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.InteractiveSession()

        # Run the Op to initialize the variables.
        sess.run(init)


        # Create DataBuffer for managing batches of train set
        data_buffer = DataBuffer(images=X_train, labels=y_train, batch_size=batch_size)






        # TensorBoard
        print('Saving graph to: %s' % log_dir)
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_dir + '/test')

        with tf.name_scope("accuracy"):
            acc_var = tf.Variable(0.0)
            tf.summary.scalar("accuracy", acc_var, collections=['train', 'test'])

        with tf.name_scope("xent"):
            xent_var = tf.Variable(0.0)
            tf.summary.scalar("xent", xent_var, collections=['train', 'test'])


        merged_summ_training = tf.summary.merge_all('train')
        merged_summ_test = tf.summary.merge_all('test')



        # Code for testing the test set in batches (normally too large)
        test_batch_size = 1000
        test_data_buffer = DataBuffer(images=X_test, labels=y_test, batch_size=test_batch_size)
        test_batch_num = int(math.ceil(len(y_test) / test_batch_size))
        test_eval_freq = 10



        # Eval test
        def eval_test_set(step, conf_matrix=False):
            final_test_accuracy, test_loss = 0,0
            y_pred, y_true = [], []
            for i in range(test_batch_num):
                images_batch, labels_batch = test_data_buffer.next_batch(shuffle_data=False)
                feed_dict_test = {images_pl: images_batch, labels_pl: labels_batch, keep_prob: 1, phase_train: False}
                batch_loss, batch_correct_predictions, batch_predictions = \
                    sess.run([loss, tf.reduce_sum(correct_predictions), predictions], feed_dict=feed_dict_test)
                test_loss += batch_loss
                final_test_accuracy += batch_correct_predictions

                if conf_matrix:
                    y_pred.extend(batch_predictions)
                    y_true.extend(labels_batch)

            final_test_accuracy /= test_size
            test_loss /= test_batch_num
            summ_test = sess.run(merged_summ_test, {xent_var: test_loss, acc_var: final_test_accuracy})
            test_writer.add_summary(summ_test, step)

            if conf_matrix:
                cm = ConfusionMatrix(y_true, y_pred)
                return final_test_accuracy, cm
            else:
                return final_test_accuracy






        start_time = time.time()  # Start time

        step = 0

        for epoch in range(max_epochs):

            for batch_index in range(num_batches_per_epoch):

                step = tf.train.global_step(sess, global_step)

                images_batch, labels_batch = data_buffer.next_batch()

                feed_dict_train_dropout = {images_pl: images_batch, labels_pl: labels_batch,
                                           keep_prob: train_dropout, phase_train: True}

                feed_dict_train_eval = {images_pl: images_batch, labels_pl: labels_batch,
                                        keep_prob: 1, phase_train: False}

                # Evaluate next batch before train
                if batch_index % 10 == 0:
                    train_loss, train_accuracy = sess.run([loss, accuracy], feed_dict_train_eval)

                    feed_dict_train_eval.update({xent_var: train_loss, acc_var: train_accuracy})
                    summ_train = sess.run(merged_summ_training, feed_dict_train_eval)
                    train_writer.add_summary(summ_train, step)

                    print('Time: ', str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start_time)))))
                    print('Epoch %d. Batch index %d, training accuracy %g' % (epoch, batch_index, train_accuracy*100))
                    # print('Epoch %d. Batch index %d, Loss %g' % (epoch, batch_index, train_loss))


                # Train model
                train_step.run(feed_dict_train_dropout)


            # Evaluate test set frequently
            if epoch % test_eval_freq == 0:
                test_accuracy = eval_test_set(step)
                print('---------------')
                print('Epoch %d. test accuracy %g' % (epoch, test_accuracy*100))
                print('---------------\n')



        # Final evaluation of the test set
        final_test_accuracy, conf_matrix = eval_test_set(step, conf_matrix=True)






        train_writer.close()
        test_writer.close()
        save_path = saver.save(sess, log_dir + '-model-' + str(patch_size) + '.ckpt')
        sess.close()
        return save_path, final_test_accuracy*100, conf_matrix
