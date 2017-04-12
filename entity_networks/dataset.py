"Define a dataset class for a single bAbI task."
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import tensorflow as tf

class Dataset(object):
    "Define a dataset class for a single bAbI task."

    def __init__(self, dataset_path, batch_size):
        self._dataset_dir = os.path.dirname(dataset_path)
        self._batch_size = batch_size

        with open(dataset_path) as file_handle:
            metadata = json.load(file_handle)

        self._dataset_size = metadata['dataset_size']
        self._max_sentence_length = metadata['max_sentence_length']
        self._max_story_length = metadata['max_story_length']
        self._max_query_length = metadata['max_query_length']
        self._dataset_size = metadata['dataset_size']
        self._vocab_size = metadata['vocab_size']
        self._tokens = metadata['tokens']
        self._datasets = metadata['datasets']

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def steps_per_epoch(self):
        "Return the number of steps per epoch for the current batch size."
        return self._dataset_size / self._batch_size

    def get_input_fn(self, dataset_name, num_epochs, shuffle):
        "Return an input function to be used with `tf.contrib.learn.Experiment`."

        def _input_fn():
            story_feature = tf.FixedLenFeature(
                shape=[self._max_story_length, self._max_sentence_length],
                dtype=tf.int64)
            query_feature = tf.FixedLenFeature(
                shape=[1, self._max_query_length],
                dtype=tf.int64)
            answer_feature = tf.FixedLenFeature(
                shape=[],
                dtype=tf.int64)

            features = {
                "story": story_feature,
                "query": query_feature,
                "answer": answer_feature,
            }

            dataset_path = os.path.join(self._dataset_dir, self._datasets[dataset_name])

            features = tf.contrib.learn.read_batch_record_features(
                file_pattern=dataset_path,
                features=features,
                batch_size=self._batch_size,
                randomize_input=shuffle,
                num_epochs=num_epochs)

            story = features['story']
            query = features['query']
            answer = features['answer']

            return {'story': story, 'query': query}, answer

        return _input_fn
