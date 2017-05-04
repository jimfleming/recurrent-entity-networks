"""
Serving input function definition.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def generate_serving_input_fn(metadata):
    "Returns _serving_input_fn for use with an export strategy."
    max_story_length = metadata['max_story_length']
    max_sentence_length = metadata['max_sentence_length']
    max_query_length = metadata['max_query_length']

    def _serving_input_fn():
        story_placeholder = tf.placeholder(
            shape=[max_story_length, max_sentence_length],
            dtype=tf.int64,
            name='story')
        query_placeholder = tf.placeholder(
            shape=[1, max_query_length],
            dtype=tf.int64,
            name='query')

        feature_placeholders = {
            'story': story_placeholder,
            'query': query_placeholder
        }

        features = {
            key: tf.expand_dims(tensor, axis=0)
            for key, tensor in feature_placeholders.items()
        }

        input_fn_ops = tf.contrib.learn.utils.input_fn_utils.InputFnOps(
            features=features,
            labels=None,
            default_inputs=feature_placeholders)

        return input_fn_ops

    return _serving_input_fn
