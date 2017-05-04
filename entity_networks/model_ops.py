"Utilities for model construction."
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import numpy as np
import tensorflow as tf

def count_parameters():
    "Count the number of parameters listed under TRAINABLE_VARIABLES."
    num_parameters = sum([np.prod(tvar.get_shape().as_list())
                          for tvar in tf.trainable_variables()])
    return num_parameters

def get_sequence_length(sequence, scope=None):
    "Determine the length of a sequence that has been padded with zeros."
    with tf.variable_scope(scope, 'SequenceLength'):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=[-1]))
        length = tf.cast(tf.reduce_sum(used, reduction_indices=[-1]), tf.int32)
        return length

def cyclic_learning_rate(
        learning_rate_min,
        learning_rate_max,
        step_size,
        global_step,
        mode='triangular',
        scope=None):
    with tf.variable_scope(scope, 'CyclicLearningRate'):
        cycle = tf.floor(1 + tf.to_float(global_step) / (2 * step_size))

        if mode == 'triangular':
            scale = 1
        elif mode == 'triangular2':
            scale = 2**(cycle - 1)
        else:
            raise ValueError('Unrecognized mode: {}'.format(mode))

        x = tf.abs(tf.to_float(global_step) / step_size - 2 * cycle + 1)
        lr = learning_rate_min + (learning_rate_max - learning_rate_min) * \
            tf.maximum(0.0, 1 - x) / scale

        return lr

def prelu(features, alpha, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU'):
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg
