from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

MAX_SENTENCE_LENGTH = 7
MAX_STORY_LENGTH = 10
MAX_QUERY_LENGTH = 4

DATASET_SIZE = 10000
VOCAB_SIZE = 22

def record_reader(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    features = tf.parse_single_example(serialized, features={
        "story": tf.FixedLenFeature([MAX_STORY_LENGTH, MAX_SENTENCE_LENGTH], dtype=tf.int64),
        "query": tf.FixedLenFeature([1, MAX_QUERY_LENGTH], dtype=tf.int64),
        "answer": tf.FixedLenFeature([], dtype=tf.int64),
    })

    story = features['story']
    query = features['query']
    answer = features['answer']

    return story, query, answer

class Dataset(object):

    def __init__(self, filename, batch_size, shuffle=False):
        self._batch_size = batch_size

        filename_queue = tf.train.string_input_producer([filename], shuffle=shuffle)
        records = record_reader(filename_queue)

        min_after_dequeue = DATASET_SIZE
        capacity = min_after_dequeue + 100 * batch_size

        if shuffle:
            self._story_batch, self._query_batch, self._answer_batch = \
                tf.train.shuffle_batch(records,
                    batch_size=batch_size,
                    min_after_dequeue=min_after_dequeue,
                    capacity=capacity)
        else:
            self._story_batch, self._query_batch, self._answer_batch = \
                tf.train.batch(records,
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
        return MAX_SENTENCE_LENGTH

    @property
    def max_story_length(self):
        return MAX_STORY_LENGTH

    @property
    def max_query_length(self):
        return MAX_QUERY_LENGTH

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    @property
    def size(self):
        return DATASET_SIZE

    @property
    def num_batches(self):
        return DATASET_SIZE // self._batch_size
