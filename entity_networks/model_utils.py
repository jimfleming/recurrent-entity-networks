from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

def count_parameters_in_scope(scope=None):
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    return np.sum([np.prod(var.get_shape().as_list()) for var in variables])
