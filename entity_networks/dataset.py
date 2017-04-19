"Define a task class for a single bAbI task."
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import tensorflow as tf

class Dataset(object):
    "Define a task class for a single bAbI task."

    def __init__(self, metadata_path, batch_size):
        self._data_dir = os.path.dirname(metadata_path)
        self._batch_size = batch_size

        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
        
        print(metadata)

        self._task_id = metadata['task_id']
        self._task_name = metadata['task_name']
        self._task_title = metadata['task_title']
        self._task_size = metadata['task_size']
        self._max_query_length = metadata['max_query_length']
        self._max_story_length = metadata['max_story_length']
        self._max_sentence_length = metadata['max_sentence_length']
        self._vocab = metadata['vocab']
        self._vocab_size = metadata['vocab_size']
        self._filenames = metadata['filenames']

    @property
    def task_id(self):
        "Return the short task id for this task."
        return self._task_id

    @property
    def task_name(self):
        "Return the task name."
        return self._task_name

    @property
    def task_title(self):
        "Return the task title."
        return self._task_title

    @property
    def task_size(self):
        "Return the size of the training set for this task."
        return self._task_size

    @property
    def max_query_length(self):
        "Return the maximum query length for this task."
        return self._max_query_length

    @property
    def max_story_length(self):
        "Return the maximum story length for this task."
        return self._max_story_length

    @property
    def max_sentence_length(self):
        "Return the maximum sentence length for this task."
        return self._max_sentence_length

    @property
    def vocab(self):
        "Return the vocab for this task."
        return self._vocab

    @property
    def vocab_size(self):
        "Return the size of the vocab for this task."
        return self._vocab_size

    @property
    def steps_per_epoch(self):
        "Return the number of steps per epoch for the current batch size."
        return self._task_size / self._batch_size

    def get_serving_input_fn(self):
        "Return an input function suitable for SavedModel."
        def _input_fn():
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
        return _input_fn

    def get_input_fn(self, task_name, num_epochs, shuffle):
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

            metadata_path = os.path.join(self._data_dir, self._filenames[task_name])

            features = tf.contrib.learn.read_batch_record_features(
                file_pattern=metadata_path,
                features=features,
                batch_size=self._batch_size,
                randomize_input=shuffle,
                num_epochs=num_epochs)

            story = features['story']
            query = features['query']
            answer = features['answer']

            return {'story': story, 'query': query}, answer

        return _input_fn
