import os, cv2
import numpy as np
import tensorflow as tf

from scipy.spatial import distance

from MLP import *
from utils import *

from Define import *

mnist = MNIST_Download("../DB/MNIST/", False)
iter_count = mnist.test.num_examples // BATCH_SIZE

# save
#img = mnist.test.next_batch(1)[0] * 255
#cv2.imwrite('search_img.jpg', img.reshape((28, 28)).astype(np.uint8))

# load
search_img = cv2.imread('search_img.jpg')
search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
search_img = search_img.reshape((1, 28 * 28)) / 255

input_var = tf.placeholder(dtype = tf.float32, shape = [None, IMAGE_SIZE])
embs = MLP(input_var)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './model/MLP.ckpt')

    search_emb = sess.run(embs, feed_dict = {input_var : search_img})
    
    for iter in range(iter_count):
        batch_x, batch_y = mnist.test.next_batch(BATCH_SIZE)

        embs = sess.run(embs, feed_dict = {input_var : batch_x})
        
        for label, emb in zip(batch_y, embs):
            dist = distance.euclidean(emb, search_emb)
            print(label, dist)
        print()
        input()
        
