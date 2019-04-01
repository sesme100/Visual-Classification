
import tensorflow as tf

from Define import *

'''
https://www.tensorflow.org/api_docs/python/tf/layers/dense

tf.layers.dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
)
'''
def MLP(x):
    x = tf.layers.dense(inputs = x, units = 512, name = 'fc_1')
    x = tf.nn.relu(x, name = 'relu_1')
    
    x = tf.layers.dense(inputs = x, units = 256, name = 'fc_2')
    x = tf.nn.relu(x, name = 'relu_2')
    
    x = tf.layers.dense(inputs = x, units = EMBEDDING_SIZE, name = 'fc_3')
    x = tf.nn.l2_normalize(x, name = 'l2_norm', dim = 1)
    print(x)

    return x
