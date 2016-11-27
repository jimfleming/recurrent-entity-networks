from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def prelu(features, initializer=None, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU', initializer=initializer):
        alpha = tf.get_variable('alpha',
            shape=features.get_shape().as_list()[1:])
        pos = tf.nn.relu(features)
        neg = tf.clip_by_value(alpha, 0, 1) * (features - tf.abs(features)) * 0.5 # TODO: Clipping may not be necessary
        return pos + neg
