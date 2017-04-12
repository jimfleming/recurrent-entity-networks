from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

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
        scope=None):
    "Define a cyclic learning rate: https://arxiv.org/abs/1506.01186"
    with tf.variable_scope(scope, 'CyclicLearningRate'):
        cycle = tf.floor(1 + tf.to_float(global_step) / (2 * step_size))
        x = tf.abs(tf.to_float(global_step) / step_size - 2 * cycle + 1)
        lr = learning_rate_min + (learning_rate_max - learning_rate_min) * \
            tf.maximum(0.0, 1 - x) / (2**(cycle - 1))
        return lr
