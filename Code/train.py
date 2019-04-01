
'''
Copyright (C) 2018 Dec 15 By JSH all rights reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Sanghyeon Jo <josanghyeokn@gmail.com>
'''

import os, cv2
import numpy as np
import tensorflow as tf

from MLP import *
from utils import *

from triplet_loss import *
from Define import *

mnist = MNIST_Download("../DB/MNIST/", False)
iter_count = mnist.train.num_examples // BATCH_SIZE

input_var = tf.placeholder(dtype = tf.float32, shape = [None, IMAGE_SIZE])
label_var = tf.placeholder(dtype = tf.int32, shape = [None])

embs = MLP(input_var)

loss_op = batch_hard_triplet_loss(label_var, embs, LOSS_ALPHA)
train_op = tf.train.AdamOptimizer(learning_rate=LEARNINT_RATE).minimize(loss_op)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train
    for epoch in range(MAX_EPOCHS):

        losses = []
        for iter in range(iter_count):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)

            # sess.run('? function', '? data')
            _, loss = sess.run([train_op, loss_op], feed_dict={input_var : batch_x,
                                                               label_var : batch_y})
            losses.append(loss)

        print("# epoch = {}, triplet_avg_loss = {}".format(epoch + 1, np.mean(losses)))
    
    # test
    saver.save(sess, './model/MLP.ckpt')

'''
Extracting ../DB/MNIST/train-images-idx3-ubyte.gz
Extracting ../DB/MNIST/train-labels-idx1-ubyte.gz
Extracting ../DB/MNIST/t10k-images-idx3-ubyte.gz
Extracting ../DB/MNIST/t10k-labels-idx1-ubyte.gz
Tensor("l2_norm:0", shape=(?, 128), dtype=float32)
# epoch = 1, triplet_avg_loss = 0.24313585460186005
# epoch = 2, triplet_avg_loss = 0.2082180380821228
# epoch = 3, triplet_avg_loss = 0.2039288580417633
# epoch = 4, triplet_avg_loss = 0.20204952359199524
# epoch = 5, triplet_avg_loss = 0.2008047103881836
# epoch = 6, triplet_avg_loss = 0.1995282620191574
# epoch = 7, triplet_avg_loss = 0.19694116711616516
# epoch = 8, triplet_avg_loss = 0.18273359537124634
# epoch = 9, triplet_avg_loss = 0.13113117218017578
# epoch = 10, triplet_avg_loss = 0.07177336513996124
# epoch = 11, triplet_avg_loss = 0.04693739488720894
# epoch = 12, triplet_avg_loss = 0.028772924095392227
# epoch = 13, triplet_avg_loss = 0.020451458171010017
# epoch = 14, triplet_avg_loss = 0.012856701388955116
# epoch = 15, triplet_avg_loss = 0.007986280135810375
# epoch = 16, triplet_avg_loss = 0.005342632532119751
# epoch = 17, triplet_avg_loss = 0.003295604372397065
# epoch = 18, triplet_avg_loss = 0.0022107951808720827
# epoch = 19, triplet_avg_loss = 0.0019070315174758434
# epoch = 20, triplet_avg_loss = 0.0010237208334729075
'''
