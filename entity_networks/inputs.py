from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def generate_input_fn(filenames, batch_size, num_epochs=None, shuffle=False):
    "Return _input_fn for use with Experiment."
    def _input_fn():
        with tf.device('/cpu:0'):
            # TODO: define inputs and labels
            # inputs = {
            #     'image': tf.constant(shape=[32, 128, 128, 3], dtype=tf.float32)
            # }
            # labels = {
            #     'label': tf.constant(shape=[32, 1], dtype=tf.int32)
            # }

            inputs = {}
            labels = {}

            return inputs, labels
    return _input_fn
