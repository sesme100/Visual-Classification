
import numpy as np
import tensorflow as tf

from Define import *

def Global_Average_Pooling(x, stride=1) :
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]

    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) 

def BatchNormalization(x, training, scope):
    with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : tf.contrib.layers.batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : tf.contrib.layers.batch_norm(inputs=x, is_training=training, reuse=True))

def residual_block_first(x, out_channel, layer_name, isTraining):
    conv_index = 1
    
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = layer_name + '_pool_1')
    x = tf.layers.conv2d(inputs = x, filters = out_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_' + str(conv_index))
    conv_index += 1

    last_x = x

    x = tf.layers.conv2d(inputs = x, filters = out_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_' + str(conv_index))
    conv_index += 1
    
    x = BatchNormalization(x, isTraining, layer_name + '_bn_1')
    x = tf.nn.relu(x, name = layer_name + '_relu_1')
    x = tf.layers.conv2d(inputs = x, filters = out_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_' + str(conv_index))
    conv_index += 1

    x = BatchNormalization(x, isTraining, layer_name + '_bn_2')
    
    x = x + last_x
    x = tf.nn.relu(x, name = layer_name + '_relu_2')

    return x

def residual_block(x, layer_name, isTraining):
    num_channel = x.get_shape().as_list()[-1]

    last_x = x
    x = tf.layers.conv2d(inputs = x, filters = num_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_1')
    x = BatchNormalization(x, isTraining, layer_name + '_bn_1')
    x = tf.nn.relu(x, name = layer_name + '_relu_1')
    x = tf.layers.conv2d(inputs = x, filters = num_channel, kernel_size = [3, 3], strides = 1, padding='same', name= layer_name + '_conv_2')
    x = BatchNormalization(x, isTraining, layer_name + '_bn_2')

    x = x + last_x
    x = tf.nn.relu(x, name = layer_name + '_relu_2')

    return x

def ResNet(input, training_flag, num_residual = 3):
    x = input

    x = tf.layers.conv2d(inputs = x, filters = CONV_FILTERS[0], kernel_size = [7, 7], strides = 2, padding = 'SAME', name = 'resnet_1_conv_1')
    x = BatchNormalization(x, training_flag, 'resnet_1_bn_1')
    x = tf.nn.relu(x, name = 'resnet_1_relu_1')
    #x = tf.layers.max_pooling2d(inputs = x, pool_size = [3, 3], strides = 2, name = 'resnet_1_pool_1')
    print(x)
    
    resnet_index = 2

    for i in range(num_residual):
        x = residual_block(x, layer_name='resnet_' + str(resnet_index), isTraining = training_flag)
        resnet_index += 1
    print(x)

    x = residual_block_first(x, CONV_FILTERS[1], layer_name='resnet_block_1', isTraining = training_flag)
    for i in range(num_residual):
        x = residual_block(x, layer_name='resnet_' + str(resnet_index), isTraining = training_flag)
        resnet_index += 1
    print(x)

    x = residual_block_first(x, CONV_FILTERS[2], layer_name='resnet_block_2', isTraining = training_flag)
    for i in range(num_residual):
        x = residual_block(x, layer_name='resnet_' + str(resnet_index), isTraining = training_flag)
        resnet_index += 1
    print(x)

    x = residual_block_first(x, CONV_FILTERS[3], layer_name='resnet_block_3', isTraining = training_flag)
    for i in range(num_residual):
        x = residual_block(x, layer_name='resnet_' + str(resnet_index), isTraining = training_flag)
        resnet_index += 1
    print(x)
    
    x = Global_Average_Pooling(x)
    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(inputs = x, units = CLASSES, name = 'fc')
        
    return x
