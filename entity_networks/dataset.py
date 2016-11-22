from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

class Dataset(object):

    def __init__(self, filenames, batch_size, shuffle=False):
        self._batch_size = batch_size
        self._size = 10000
        self._vocab_size = 22
        self._max_sentence_length = 7
        self._max_story_length = 10
        self._max_query_length = 4

        filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
        capacity = self._size + 100 * batch_size

        reader = tf.TFRecordReader()
        _, serialized = reader.read(filename_queue)
        sequence_features = {
            "story": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "query": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }
        context_features = {
            "answer": tf.FixedLenFeature([], dtype=tf.int64),
        }
        context_features, sequence_features = tf.parse_single_sequence_example(serialized,
            context_features=context_features,
            sequence_features=sequence_features)

        story = sequence_features['story']
        query = sequence_features['query']
        answer = context_features['answer']

        story.set_shape([self._max_story_length * self._max_sentence_length])
        query.set_shape([self._max_query_length])

        story = tf.reshape(story, [self._max_story_length, self._max_sentence_length])
        query = tf.reshape(query, [1, self._max_query_length])

        if shuffle:
            self._story_batch, self._query_batch, self._answer_batch = \
                tf.train.shuffle_batch([story, query, answer],
                    batch_size=batch_size,
                    min_after_dequeue=self._size,
                    capacity=capacity)
        else:
            self._story_batch, self._query_batch, self._answer_batch = \
                tf.train.batch([story, query, answer],
                    batch_size=batch_size,
                    capacity=capacity)

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
        return self._size
