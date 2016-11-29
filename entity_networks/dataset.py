from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import tensorflow as tf

class Dataset(object):

    def __init__(self, dataset_path, batch_size):
        self.dataset_dir = os.path.dirname(dataset_path)
        self.batch_size = batch_size
        self.examples_per_epoch = 10000

        with open(dataset_path) as f:
            metadata = json.load(f)

        self.max_sentence_length = metadata['max_sentence_length']
        self.max_story_length = metadata['max_story_length']
        self.max_query_length = metadata['max_query_length']
        self.dataset_size = metadata['dataset_size']
        self.vocab_size = metadata['vocab_size']
        self.tokens = metadata['tokens']
        self.datasets = metadata['datasets']

    @property
    def steps_per_epoch(self):
        return self.batch_size * self.examples_per_epoch

    def get_input_fn(self, name, num_epochs, shuffle):
        def input_fn():
            features = {
                "story": tf.FixedLenFeature([self.max_story_length, self.max_sentence_length], dtype=tf.int64),
                "query": tf.FixedLenFeature([1, self.max_query_length], dtype=tf.int64),
                "answer": tf.FixedLenFeature([], dtype=tf.int64),
            }

            dataset_path = os.path.join(self.dataset_dir, self.datasets[name])
            features = tf.contrib.learn.read_batch_record_features(dataset_path,
                features=features,
                batch_size=self.batch_size,
                randomize_input=shuffle,
                num_epochs=num_epochs)

            story = features['story']
            query = features['query']
            answer = features['answer']

            return {'story': story, 'query': query}, answer
        return input_fn
