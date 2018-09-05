from sklearn.model_selection import StratifiedKFold

import CNNModel_2D
import IndianPines_Input,Salinas_Input
import DataBuffer
import Decoder
import spectral
import tensorflow as tf
import time
import numpy as np
from flask import session,flash
from PIL import Image
import math


# Train best model with all input as training to obtain output classification image



def create_test(config):

# Configurable parameters
# config = {}
# config['patch_size'] = 9
# config['kernel_size'] = 3
# config['conv1_channels'] = 32
# config['conv2_channels'] = 64
# config['fc1_units'] = 1024
# config['batch_size'] = 64
# config['max_epochs'] = 100
# config['train_dropout'] = 0.5
# config['initial_learning_rate'] = 0.01
# config['decaying_lr'] = True

    log_dir = 'app/static/data/models/' + config['image_name'] +"/"
    config['log_dir'] = log_dir




    # Input data
    if(config['image_name']=="indianpines"):
        input = IndianPines_Input.IndianPines_Input()
    else:
        input = Salinas_Input.Salinas_Input()


    X, y = input.read_data(config['patch_size'])



    tf.reset_default_graph()

    with tf.Graph().as_default():

        # Size of input
        input_size = len(X)
        num_batches_per_epoch = int(input_size / config['batch_size'])
        test_size = len(y)


        # Create placeholders
        images_pl, labels_pl = CNNModel_2D.placeholder_inputs(config['patch_size'],input.input_channels)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, keep_prob = CNNModel_2D.inference(images_pl, input.input_channels,config['patch_size'],
                                                    config['kernel_size'], config['conv1_channels'],
                                                    config['conv2_channels'], config['fc1_units'],
                                                    input.num_classes)

        # Calculate loss.
        loss = CNNModel_2D.loss(logits, labels_pl)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Define learning rate
        # Decay once per epoch , using an exponential schedule starting at initial_learning_rate
        if config['decaying_lr']:
            learning_rate = tf.train.exponential_decay(
                config['initial_learning_rate'],  # Base learning rate.
                global_step,  # Current index into the dataset.
                num_batches_per_epoch,  # Decay step.
                0.96,  # Decay rate.
                staircase=True)
        else:
            learning_rate = config['initial_learning_rate']


        train_step = CNNModel_2D.training(loss, learning_rate, global_step)

        # Add the Op to compare the logits to the labels during evaluation.
        predictions,correct_predictions,accuracy = CNNModel_2D.evaluation(logits, labels_pl)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.InteractiveSession()

        # Run the Op to initialize the variables.
        sess.run(init)

        # Create DataBuffer for managing batches of train set
        data_buffer = DataBuffer.DataBuffer(images=X, labels=y, batch_size=config['batch_size'])




        # TensorBoard
        train_writer = tf.summary.FileWriter(log_dir + str(config['id']) + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_dir + str(config['id']) + '/test')

        with tf.name_scope("accuracy"):
            acc_var = tf.Variable(0.0)
            tf.summary.scalar("accuracy", acc_var, collections=['train', 'test'])

        with tf.name_scope("xent"):
            xent_var = tf.Variable(0.0)
            tf.summary.scalar("xent",xent_var,collections=['train', 'test'])


        merged_summ_training = tf.summary.merge_all('train')
        merged_summ_test = tf.summary.merge_all('test')


        # Code for testing the test set in batches (normally too large)
        test_batch_size = 1000
        test_data_buffer = DataBuffer.DataBuffer(images=X, labels=y, batch_size=test_batch_size)
        test_batch_num = int(math.ceil(len(y) / test_batch_size))
        test_eval_freq = 5




        def eval_test_set(step,conf_matrix=False):
            final_test_accuracy,test_loss = 0,0
            y_pred,y_true = [],[]
            for i in range(test_batch_num):
                images_batch, labels_batch = test_data_buffer.next_batch(shuffle_data=False)
                feed_dict_test = {images_pl: images_batch, labels_pl: labels_batch, keep_prob: 1}
                batch_loss,batch_correct_predictions,batch_predictions = \
                    sess.run([loss,tf.reduce_sum(correct_predictions),predictions],feed_dict= feed_dict_test)
                test_loss += batch_loss
                final_test_accuracy += batch_correct_predictions


            final_test_accuracy /= test_size
            test_loss /= test_batch_num
            summ_test = sess.run(merged_summ_test, {xent_var: test_loss, acc_var: final_test_accuracy})
            test_writer.add_summary(summ_test, step)

            return final_test_accuracy












        start_time = time.time()  # Start time

        session['epoch'] = 0
        flash(0)

        for epoch in range(config['max_epochs']):

            session['epoch'] = epoch
            print("Session epoch:" + str(session['epoch']))
            flash(epoch)


            for batch_index in range(num_batches_per_epoch + 1):

                step = tf.train.global_step(sess, global_step)

                images_batch, labels_batch = data_buffer.next_batch()

                feed_dict_train_dropout = {images_pl: images_batch, labels_pl: labels_batch, keep_prob: config['train_dropout']}
                feed_dict_train_eval = {images_pl: images_batch, labels_pl: labels_batch, keep_prob: 1}

                # Evaluate next batch before train
                train_accuracy = accuracy.eval(feed_dict_train_eval) * 100



                if batch_index % 10 == 0:

                    train_loss,train_accuracy = sess.run([loss,accuracy],feed_dict_train_eval)
                    feed_dict_train_eval.update({xent_var: train_loss,acc_var:train_accuracy})
                    summ_train = sess.run(merged_summ_training, feed_dict_train_eval)
                    train_writer.add_summary(summ_train, step)


                    #percent_advance = str(batch_index * 100 / num_batches_per_epoch)
                    print('Time: ', str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start_time)))))
                    print('Epoch %d. Batch index %d, training accuracy %g' % (epoch, batch_index, train_accuracy))
                    print('---------------\n')


                # Train model
                train_step.run(feed_dict_train_dropout)

            if(epoch % test_eval_freq == 0):
                test_accuracy = eval_test_set(step)
                print('---------------')
                print('Epoch %d. test accuracy %g' % (epoch, test_accuracy*100))
                print('---------------\n')



        train_writer.close()
        test_writer.close()
        save_path = saver.save(sess, log_dir + 'model-' + str(config['id']) + '.ckpt')





    predicted_image,final_accuracy = Decoder.decode(input, config, save_path)

    image_path = 'app/static/data/images/outputmap_'+ str(config['image_name']) + "_" + str(config['id'])

    #ground_truth = spectral.imshow(classes = input.target_data,figsize =(9,9))
    #predict_image = spectral.imshow(classes = predicted_image.astype(int),figsize =(9,9))
    #spectral.save_rgb('gt.png', input.target_data,colors=spectral.spy_colors, format='png')
    spectral.save_rgb(image_path +".png", predicted_image,colors=spectral.spy_colors)

    img = Image.open(image_path +".png")
    img = img.resize((700,700), resample= Image.ANTIALIAS)
    img.save(image_path +"Big.png")

    return predicted_image,final_accuracy
