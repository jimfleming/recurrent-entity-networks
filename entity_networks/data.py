from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def record_reader(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    sequence_features = {
        "story": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "query": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }
    context_features = {
        "story_length": tf.FixedLenFeature([], dtype=tf.int64),
        "query_length": tf.FixedLenFeature([], dtype=tf.int64),
        "answer": tf.FixedLenFeature([], dtype=tf.int64),
    }
    context_features, sequence_features = tf.parse_single_sequence_example(
        serialized=serialized,
        context_features=context_features,
        sequence_features=sequence_features)
    story_length = context_features['story_length']
    query_length = context_features['query_length']
    answer = context_features['answer']
    story = sequence_features['story']
    query = sequence_features['query']
    return story, query, answer, story_length, query_length

def input_pipeline(filenames, batch_size, num_epochs=None, shuffle=False):
    filename_queue = tf.train.string_input_producer(filenames,
        num_epochs=num_epochs,
        shuffle=shuffle)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 10 * batch_size
    records = record_reader(filename_queue)
    batches = tf.train.batch(
        tensors=records,
        batch_size=batch_size,
        capacity=capacity,
        dynamic_pad=False)
    return batches
