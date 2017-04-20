from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def serving_input_fn():
    story = tf.zeros(
        shape=[1, self._max_story_length, self._max_sentence_length],
        dtype=tf.int64)
    query = tf.zeros(
        shape=[1, 1, self._max_query_length],
        dtype=tf.int64)
    features = {
        'story': story,
        'query': query
    }
    input_fn_ops = tf.contrib.learn.InputFnOps(
        features=features,
        labels=None,
        default_inputs=features)
    return input_fn_ops
