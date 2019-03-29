
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

def bottleneck_layer(input, layer_name, isTraining):
    x = BatchNormalization(input, isTraining, layer_name + '_bn_1')
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(inputs = x, filters = K * 4, kernel_size = [1, 1], strides = 1, padding = 'SAME', name = layer_name + '_conv_1')
    x = tf.layers.dropout(inputs = x, rate = DROP_OUT, training = isTraining)

    x = BatchNormalization(x, isTraining, layer_name + '_bn_2')
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(inputs = x, filters = K, kernel_size = [3, 3], strides = 1, padding = 'SAME', name = layer_name + '_conv_2')
    x = tf.layers.dropout(inputs = x, rate = DROP_OUT, training = isTraining)

    return x

def transition_layer(input, layer_name, isTraining):
    x = BatchNormalization(input, isTraining, layer_name + '_bn_1')
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(inputs = x, filters = K, kernel_size = [1, 1], strides = 1, padding = 'SAME', name = layer_name + '_conv_1')
    x = tf.layers.dropout(inputs = x, rate = DROP_OUT, training = isTraining)
    x = tf.layers.average_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = layer_name + '_pool_1')
    return x

'''
[224, 224, 3]
[224, 224, 6]
[224, 224, 12]
[224, 224, 24]

[224, 224, 24 + 12 + 6 + 3]
'''
def Concatenation(feature_maps):
    x = feature_maps[0]

    for i in range(len(feature_maps) - 1):
        x = tf.concat([x, feature_maps[i]], axis = -1)

    return x

'''
K * 4 * LAYERS
'''
def DenseBlock(input, layers, layers_name, isTraining):
    feature_maps = []
    feature_maps.append(input)

    x = bottleneck_layer(input = input, layer_name = layers_name + '_bottle_' + str(0), isTraining = isTraining)
    feature_maps.append(x)

    for i in range(layers - 1):
        x = Concatenation(feature_maps = feature_maps)

        x = bottleneck_layer(input = x, layer_name = layers_name + '_bottle_' + str(i + 1), isTraining = isTraining)
        feature_maps.append(x)

    x = Concatenation(feature_maps)
    return x

def DenseNet(input, training_flag):
    x = input

    x = tf.layers.conv2d(inputs = x, filters = K * 2, kernel_size = [7, 7], strides = 2, padding = 'SAME', name = 'first_conv_1')
    # x = tf.layers.max_pooling2d(inputs = x, kernel_size = [3, 3], strides = 2)
    print(x)

    x = DenseBlock(input = x, layers = LAYERS[0], layers_name = 'dense_1', isTraining = training_flag)
    x = transition_layer(input = x, layer_name = 'trans_1', isTraining = training_flag)
    print(x)

    x = DenseBlock(input = x, layers = LAYERS[1], layers_name = 'dense_2', isTraining = training_flag)
    x = transition_layer(input = x, layer_name = 'trans_2', isTraining = training_flag)
    print(x)

    x = DenseBlock(input = x, layers = LAYERS[2], layers_name = 'dense_3', isTraining = training_flag)
    x = transition_layer(input = x, layer_name = 'trans_3', isTraining = training_flag)
    print(x)

    x = DenseBlock(input = x, layers = LAYERS[3], layers_name = 'dense_4', isTraining = training_flag)
    print(x)

    x = BatchNormalization(x, training_flag, 'bn')
    x = tf.nn.relu(x)
    x = Global_Average_Pooling(x)
    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(inputs = x, units = CLASSES, name = 'fc')
    print(x)

    return x

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, 32, 32, 3])
    training_var = tf.placeholder(tf.bool)
    output = DenseNet(input_var, training_var)
    print(output)
