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
        "answer": tf.FixedLenFeature([], dtype=tf.int64),
    }
    context_features, sequence_features = tf.parse_single_sequence_example(serialized,
        context_features=context_features,
        sequence_features=sequence_features)
    answer = context_features['answer']
    story = sequence_features['story']
    query = sequence_features['query']

    # TODO: Turn both functions into class methods with public properties
    # to the story, query and answer as well as other vars.
    max_sentence_length = 7
    max_story_length = 10
    max_query_length = 4

    story.set_shape([max_story_length * max_sentence_length])
    query.set_shape([max_query_length])

    story = tf.reshape(story, [max_story_length, max_sentence_length])
    query = tf.reshape(query, [1, max_query_length])

    return story, query, answer

def input_pipeline(filenames, batch_size, num_epochs=None, shuffle=False):
    filename_queue = tf.train.string_input_producer(filenames,
        num_epochs=num_epochs,
        shuffle=shuffle)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 10 * batch_size
    records = record_reader(filename_queue)
    batches = tf.train.shuffle_batch(records,
        batch_size=batch_size,
        min_after_dequeue=min_after_dequeue,
        capacity=capacity)
    return batches
