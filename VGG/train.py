import os
import sys
import cv2
import numpy as np

import pickle
import tensorflow as tf

from time import time

from VGG16 import *
from utils import *

from Define import *

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels

def load_data(files, data_dir, label_count):
    imgs = []
    labels = []

    for file in files:
        dic = unpickle(data_dir + file)

        image_cnt = len(dic[b'filenames'])
        for i in range(image_cnt):
            label = dic[b'labels'][i]
            filename = dic[b'filenames'][i]
            data = dic[b'data'][i]
            
            image_size = 32
            one_channel_size = image_size * image_size

            r = data[:one_channel_size]
            g = data[one_channel_size : one_channel_size * 2]
            b = data[one_channel_size * 2 : ]

            r = r.reshape((image_size, image_size)).astype(np.uint8)
            g = g.reshape((image_size, image_size)).astype(np.uint8)
            b = b.reshape((image_size, image_size)).astype(np.uint8)

            img = cv2.merge((b, g, r))

            imgs.append(img)
            labels.append(one_hot(label, CLASSES))

    return np.asarray(imgs), np.asarray(labels)

def shuffle(imgs, labels):
    indexs = np.arange(len(imgs))
    np.random.shuffle(indexs)

    imgs = imgs[indexs]
    labels = labels[indexs]
    return imgs, labels

if __name__ == '__main__':

    dataset_dir = '../../DB/CIFAR/batch/'

    meta = unpickle(dataset_dir + 'batches.meta')
    label_names = meta[b'label_names']
    label_count = len(label_names)

    #for label_name in label_names:
    #    print(str(label_name))

    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_imgs, train_labels = load_data(train_files, dataset_dir, label_count)
    test_imgs, test_labels = load_data(['test_batch'], dataset_dir, label_count)

    train_imgs, train_labels = shuffle(train_imgs, train_labels)
    test_imgs, test_labels = shuffle(test_imgs, test_labels)

    print('# dataset batch load')

    train_length = int(len(train_imgs) * 0.9)
    valid_imgs = train_imgs[train_length:]
    valid_labels = train_labels[train_length:]

    train_imgs = train_imgs[:train_length]
    train_labels = train_labels[:train_length]

    train_imgs = train_imgs / 127.5 - 1
    valid_imgs = valid_imgs / 127.5 - 1
    test_imgs = test_imgs / 127.5 - 1

    print('# prepare data')
    
    #path define
    model_dir = './model/'
    model_name = 'vgg16_{}.ckpt'
    model_path = model_dir + model_name
    
    #model build
    input_var = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], name = 'input')
    label_var = tf.placeholder(tf.float32, shape=[None, CLASSES])
    training_var = tf.placeholder(tf.bool)

    vgg = VGG16(input_var, training_var)
    print('# model build')

    learning_rate_var = tf.placeholder(tf.float32, name = 'learning_rate')
    
    #loss
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label_var, logits = vgg))

    #optimizer
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate = learning_rate_var, epsilon = 1e-4).minimize(loss_op)

    #accuracy
    correct_prediction = tf.equal(tf.argmax(vgg, 1), tf.argmax(label_var, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #save
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #epoch
        print('# set batch size :', BATCH_SIZE)

        best_valid_epoch = 0
        best_valid_accuracy = 0.0

        train_start_time = time()
        batch_acc_list = []
        batch_loss_list = []

        iter_learning_rate = LEARNING_RATE
        for iter in range(1, MAX_ITERS + 1):
            if iter == (MAX_ITERS * 0.5) or iter == (MAX_ITERS * 0.75):
                iter_learning_rate /= 10
            
            #init
            train_imgs, train_labels = shuffle(train_imgs, train_labels)

            batch_x = train_imgs[:BATCH_SIZE]
            batch_y = train_labels[:BATCH_SIZE]

            input_map = { input_var : batch_x,
                          label_var : batch_y,
                          learning_rate_var : iter_learning_rate,
                          training_var : True }
            
            _, batch_loss = sess.run([train_op, loss_op], feed_dict = input_map)
            batch_acc = accuracy_op.eval(feed_dict = input_map)

            batch_acc_list.append(batch_acc)
            batch_loss_list.append(batch_loss)

            #log
            if iter % 100 ==0:
                train_time = time() - train_start_time

                batch_acc = np.mean(batch_acc_list)
                batch_loss = np.mean(batch_loss_list)

                print('# iter : {}, loss : {:.4f}, accuracy : {:.4f}, time : {}'.format(iter, batch_loss,
                                                                                                batch_acc,
                                                                                                train_time))
                train_start_time = time()
                batch_acc_list = []
                batch_loss_list = []

            #validation
            if iter % 1000 == 0:

                batch_acc_list = []
                max_batch_index = int(len(valid_imgs) / BATCH_SIZE)

                valid_start_time = time()
                
                for i in range(max_batch_index):
                    batch_x = valid_imgs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                    batch_y = valid_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

                    input_map = { input_var : batch_x,
                                  label_var : batch_y,
                                  training_var : False }

                    batch_acc = accuracy_op.eval(feed_dict = input_map)
                    batch_acc_list.append(batch_acc)

                valid_time = time() - valid_start_time

                valid_accuracy = np.mean(batch_acc_list)

                if best_valid_accuracy < valid_accuracy:
                    best_valid_accuracy = valid_accuracy
                    best_valid_iter = iter

                    #save
                    saver.save(sess, model_path.format(iter))

                print('# iter {} valid set accuracy : {:.4f}, best valid accuracy : {:.4f}, time : {}'.format(iter, valid_accuracy, best_valid_accuracy, valid_time))

        #test
        saver.restore(sess, model_path.format(best_valid_iter))

        batch_acc_list = []
        max_batch_index = int(len(test_imgs) / BATCH_SIZE)
                
        for i in range(max_batch_index):
            batch_x = test_imgs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_y = test_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            input_map = { input_var : batch_x,
                          label_var : batch_y,
                          training_var : False }

            batch_acc = accuracy_op.eval(feed_dict = input_map)
            batch_acc_list.append(batch_acc)

        test_accuracy = np.mean(batch_acc_list)
        print('# test set accuracy : {:.4f}'.format(test_accuracy))
