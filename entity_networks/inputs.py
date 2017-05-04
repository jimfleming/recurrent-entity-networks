"""
Module responsible for input data.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def generate_input_fn(filename, metadata, batch_size, num_epochs=None, shuffle=False):
    "Return _input_fn for use with Experiment."
    def _input_fn():
        max_story_length = metadata['max_story_length']
        max_sentence_length = metadata['max_sentence_length']
        max_query_length = metadata['max_query_length']

        with tf.device('/cpu:0'):
            story_feature = tf.FixedLenFeature(
                shape=[max_story_length, max_sentence_length],
                dtype=tf.int64)
            query_feature = tf.FixedLenFeature(
                shape=[1, max_query_length],
                dtype=tf.int64)
            answer_feature = tf.FixedLenFeature(
                shape=[],
                dtype=tf.int64)

            features = {
                'story': story_feature,
                'query': query_feature,
                'answer': answer_feature,
            }

            record_features = tf.contrib.learn.read_batch_record_features(
                file_pattern=filename,
                features=features,
                batch_size=batch_size,
                randomize_input=shuffle,
                num_epochs=num_epochs)

            story = record_features['story']
            query = record_features['query']
            answer = record_features['answer']

            features = {
                'story': story,
                'query': query,
            }

            return features, answer

    return _input_fn
