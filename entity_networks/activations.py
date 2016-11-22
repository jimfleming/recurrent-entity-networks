from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def prelu(features, initializer=None, name=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(name or 'PReLU'):
        alpha = tf.get_variable('alpha',
            shape=features.get_shape().as_list()[1:],
            initializer=initializer)
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg
