from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json
import tensorflow as tf

class Dataset(object):

    def __init__(self, path, name, batch_size, shuffle=False, num_epochs=None, num_threads=4):
        self._batch_size = batch_size

        with open(path) as f:
            metadata = json.load(f)

        self._max_sentence_length = metadata['max_sentence_length']
        self._max_story_length = metadata['max_story_length']
        self._max_query_length = metadata['max_query_length']

        self._dataset_size = metadata['dataset_size']
        self._vocab_size = metadata['vocab_size']

        filename = metadata['datasets'][name]
        filename_queue = tf.train.string_input_producer([filename],
            num_epochs=num_epochs,
            shuffle=shuffle)
        records = [self.record_reader(filename_queue) for _ in range(num_threads)]

        min_after_dequeue = self._dataset_size
        capacity = min_after_dequeue + 100 * batch_size

        if shuffle:
            self._story_batch, self._query_batch, self._answer_batch = \
                tf.train.shuffle_batch_join(records,
                    batch_size=batch_size,
                    min_after_dequeue=min_after_dequeue,
                    capacity=capacity)
        else:
            self._story_batch, self._query_batch, self._answer_batch = \
                tf.train.batch_join(records,
                    batch_size=batch_size,
                    capacity=capacity)

    def record_reader(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized = reader.read(filename_queue)

        features = tf.parse_single_example(serialized, features={
            "story": tf.FixedLenFeature([self._max_story_length, self._max_sentence_length], dtype=tf.int64),
            "query": tf.FixedLenFeature([1, self._max_query_length], dtype=tf.int64),
            "answer": tf.FixedLenFeature([], dtype=tf.int64),
        })

        story = features['story']
        query = features['query']
        answer = features['answer']

        return story, query, answer

    @property
    def story_batch(self):
        return self._story_batch

    @property
    def query_batch(self):
        return self._query_batch

    @property
    def answer_batch(self):
        return self._answer_batch

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_sentence_length(self):
        return self._max_sentence_length

    @property
    def max_story_length(self):
        return self._max_story_length

    @property
    def max_query_length(self):
        return self._max_query_length

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def size(self):
        return self._dataset_size

    @property
    def num_batches(self):
        return self._dataset_size // self._batch_size
