
import numpy as np
import tensorflow as tf

from Define import *

def conv_relu(x, num_filter, training, layer_name):

    x = tf.layers.conv2d(inputs = x, filters = num_filter, kernel_size = [3, 3], strides = 1, padding = 'SAME', name = layer_name + '_conv_1')
    x = tf.nn.relu(x, name = layer_name + '_relu_1')
    return x

def VGG16(input, training_flag):
    x = input
    print(x)
    
    #block 1
    for i in range(2):
        x = conv_relu(x, CONV_FILTERS[0], training_flag, 'vgg_1_' + str(i))
        print(x)
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_1_pool_1')
    print(x)
    
    #block 2
    for i in range(2):
        x = conv_relu(x, CONV_FILTERS[1], training_flag, 'vgg_2_' + str(i))
        print(x)
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_2_pool_1')
    print(x)

    #block 3
    for i in range(3):
        x = conv_relu(x, CONV_FILTERS[2], training_flag, 'vgg_3_' + str(i))
        print(x)
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_3_pool_1')
    print(x)

    #block 4
    for i in range(3):
        x = conv_relu(x, CONV_FILTERS[3], training_flag, 'vgg_4_' + str(i))
        print(x)
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_4_pool_1')
    print(x)

    #block 5
    for i in range(3):
        x = conv_relu(x, CONV_FILTERS[3], training_flag, 'vgg_5_' + str(i))
        print(x)
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'vgg_5_pool_1')
    print(x)

    # (224, 224, 3) -> (224 * 224 * 3)
    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(inputs = x, units = FC_PARAMTERS[0], name = 'fc_1')
    x = tf.layers.dropout(inputs = x, rate = DROP_OUT, training = training_flag)
    
    x = tf.layers.dense(inputs = x, units = FC_PARAMTERS[1], name = 'fc_2')
    x = tf.layers.dropout(inputs = x, rate = DROP_OUT, training = training_flag)

    x = tf.layers.dense(inputs = x, units = CLASSES, name = 'fc_3')
        
    return x

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, 224, 224, 3])
    output = VGG16(input_var, True)
    print(output)
