from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

def count_parameters_in_scope(scope=None):
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    return np.sum([np.prod(var.get_shape().as_list()) for var in variables])

def get_sequence_length(sequence, scope=None):
    """
    This is a hacky way of determining the actual length of a sequence that has been padded with zeros.
    """
    with tf.variable_scope(scope, 'SequenceLength'):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=[-1]))
        length = tf.cast(tf.reduce_sum(used, reduction_indices=[-1]), tf.int32)
        return length

def get_sequence_mask(sequence, scope=None):
    """
    This is a hacky way of masking the padded sentence embeddings.
    """
    with tf.variable_scope(scope, 'PaddingMask'):
        sequence = tf.reduce_sum(sequence, reduction_indices=[-1], keep_dims=True)
        mask = tf.to_float(tf.greater(sequence, 0))
        return tf.expand_dims(mask, -1)
