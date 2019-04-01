
import numpy as np

def MNIST_Download(download_path, _one_hot = True):
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets(download_path, one_hot=_one_hot)
    return mnist

