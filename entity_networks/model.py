from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from functools import partial

from entity_networks.activations import prelu
from entity_networks.dynamic_memory_cell import DynamicMemoryCell
from entity_networks.model_utils import get_sequence_length, get_sequence_mask

def model_fn(features, labels, params, mode, scope=None):
    vocab_size = params['vocab_size']
    num_blocks = params['num_blocks']
    embedding_size = params['embedding_size']

    story = features['story']
    query = features['query']

    initializer = tf.random_normal_initializer(stddev=0.01) # We use 1e-2 rather than 1e-1 as in the paper to avoid NaN
    activation = partial(prelu, initializer=tf.constant_initializer(1.0))

    with tf.variable_scope(scope, 'EntityNetwork', initializer=initializer):
        embedding_params = tf.get_variable('embedding_params', [vocab_size, embedding_size])

        story_embedding = tf.nn.embedding_lookup(embedding_params, story)
        query_embedding = tf.nn.embedding_lookup(embedding_params, query)

        # Mask embeddings
        story_mask = get_sequence_mask(story)
        query_mask = get_sequence_mask(query)

        # Input Module
        encoded_story = get_input_encoding(story_embedding * story_mask, initializer, 'StoryEncoding')
        encoded_query = get_input_encoding(query_embedding * query_mask, initializer, 'QueryEncoding')

        # Memory Module
        cell = DynamicMemoryCell(num_blocks, embedding_size,
            initializer=initializer,
            activation=activation)

        # Recurrence
        _, last_state = tf.nn.dynamic_rnn(cell, encoded_story,
            sequence_length=get_sequence_length(encoded_story),
            dtype=tf.float32)

        # Output Module
        output = get_output(last_state, encoded_query,
            num_blocks=num_blocks,
            vocab_size=vocab_size,
            initializer=initializer,
            activation=activation)
        prediction = tf.argmax(output, 1)

        # Summaries
        # tf.contrib.layers.summarize_activation(output)
        # tf.contrib.layers.summarize_variables(name_filter='.*/alpha')

        # Training
        loss = get_loss(output, labels)
        train_op = get_train_op(loss, params, mode)

        return prediction, loss, train_op

def get_input_encoding(embedding, initializer=None, scope=None):
    """
    Implementation of the learned multiplicative mask from Section 2.1, Equation 1. This module is also described
    in [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852) as Position Encoding (PE). The mask allows
    the ordering of words in a sentence to affect the encoding.
    """
    with tf.variable_scope(scope, 'Encoding', initializer=initializer):
        _, _, max_sentence_length, _ = embedding.get_shape().as_list()
        positional_mask = tf.get_variable('positional_mask', [max_sentence_length, 1])
        encoded_input = tf.reduce_sum(embedding * positional_mask, reduction_indices=[2])
        return encoded_input

def get_output(last_state, encoded_query, num_blocks, vocab_size, activation=tf.nn.relu, initializer=None, scope=None):
    """
    Implementation of Section 2.3, Equation 6. This module is also described in more detail here:
    [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852).
    """
    with tf.variable_scope(scope, 'Output', initializer=initializer):
        last_state = tf.pack(tf.split(1, num_blocks, last_state), axis=1)
        _, _, embedding_size = last_state.get_shape().as_list()

        # Use the encoded_query to attend over memories (hidden states of dynamic last_state cell blocks)
        attention = tf.reduce_sum(last_state * encoded_query, reduction_indices=[2])
        attention = tf.nn.softmax(attention)
        attention = tf.expand_dims(attention, 2)

        # Weight memories by attention vectors
        u = tf.reduce_sum(last_state * attention, reduction_indices=[1])

        # R acts as the decoder matrix to convert from internal state to the output vocabulary size
        R = tf.get_variable('R', [embedding_size, vocab_size])
        H = tf.get_variable('H', [embedding_size, embedding_size])

        q = tf.squeeze(encoded_query, squeeze_dims=[1])
        y = tf.matmul(activation(q + tf.matmul(u, H)), R)
        return y

def get_loss(output, labels):
    if labels is None:
        return None
    return tf.contrib.losses.sparse_softmax_cross_entropy(output, labels)

def get_train_op(loss, params, mode):
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
        return None

    clip_gradients = params['clip_gradients']
    learning_rate_init = params['learning_rate_init']
    learning_rate_decay_rate = params['learning_rate_decay_rate']
    learning_rate_decay_steps = params['learning_rate_decay_steps']

    global_step = tf.contrib.framework.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        learning_rate=learning_rate_init,
        decay_steps=learning_rate_decay_steps,
        decay_rate=learning_rate_decay_rate,
        global_step=global_step,
        staircase=True)
    tf.contrib.layers.summarize_tensor(learning_rate, tag='learning_rate')

    train_op = tf.contrib.layers.optimize_loss(loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer='Adam',
        clip_gradients=clip_gradients)
    return train_op
