from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from functools import partial

from entity_networks.activations import prelu
from entity_networks.dynamic_memory_cell import DynamicMemoryCell
from entity_networks.model_utils import get_sequence_length

def model_fn(features, labels, params, mode, scope=None):
    embedding_size = params['embedding_size']
    num_blocks = params['num_blocks']
    vocab_size = params['vocab_size']
    debug = params['debug']

    story = features['story']
    query = features['query']

    batch_size = tf.shape(story)[0]

    normal_initializer = tf.random_normal_initializer(stddev=0.1)
    ones_initializer = tf.constant_initializer(1.0)

    # PReLU activations have their alpha parameters initialized to 1
    # so they may be identity before training.
    activation = partial(prelu, initializer=ones_initializer)

    with tf.variable_scope(scope, 'EntityNetwork', initializer=normal_initializer):
        # Embeddings
        # The embedding mask forces the special "pad" embedding to zeros.
        embedding_params = tf.get_variable('embedding_params', [vocab_size, embedding_size])
        embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(vocab_size)],
            dtype=tf.float32,
            shape=[vocab_size, 1])
        embedding_params_masked = embedding_params * embedding_mask

        story_embedding = tf.nn.embedding_lookup(embedding_params_masked, story)
        query_embedding = tf.nn.embedding_lookup(embedding_params_masked, query)

        # Input Module
        encoded_story = get_input_encoding(story_embedding, ones_initializer, 'StoryEncoding')
        encoded_query = get_input_encoding(query_embedding, ones_initializer, 'QueryEncoding')

        # Memory Module
        # We define the keys outside of the cell so they may be used for state initialization.
        keys = [tf.get_variable('key_{}'.format(j), [embedding_size]) for j in range(num_blocks)]

        cell = DynamicMemoryCell(num_blocks, embedding_size, keys,
            initializer=normal_initializer,
            activation=activation)

        # Recurrence
        initial_state = cell.zero_state(batch_size, tf.float32)
        sequence_length = get_sequence_length(encoded_story)
        _, last_state = tf.nn.dynamic_rnn(cell, encoded_story,
            sequence_length=sequence_length,
            initial_state=initial_state)

        # Output Module
        output = get_output(last_state, encoded_query,
            num_blocks=num_blocks,
            vocab_size=vocab_size,
            initializer=normal_initializer,
            activation=activation)
        prediction = tf.argmax(output, 1)

        # Training
        loss = get_loss(output, labels, mode)
        train_op = get_train_op(loss, params, mode)

        if debug:
            tf.contrib.layers.summarize_tensor(sequence_length, 'sequence_length')
            tf.contrib.layers.summarize_tensor(encoded_story, 'encoded_story')
            tf.contrib.layers.summarize_tensor(encoded_query, 'encoded_query')
            tf.contrib.layers.summarize_tensor(last_state, 'last_state')
            tf.contrib.layers.summarize_tensor(output, 'output')
            tf.contrib.layers.summarize_variables()

            tf.add_check_numerics_ops()

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

def get_output(last_state, encoded_query, num_blocks, vocab_size,
        activation=tf.nn.relu,
        initializer=None,
        scope=None):
    """
    Implementation of Section 2.3, Equation 6. This module is also described in more detail here:
    [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852).
    """
    with tf.variable_scope(scope, 'Output', initializer=initializer):
        last_state = tf.pack(tf.split(1, num_blocks, last_state), axis=1)
        _, _, embedding_size = last_state.get_shape().as_list()

        # Use the encoded_query to attend over memories (hidden states of dynamic last_state cell blocks)
        attention = tf.reduce_sum(last_state * encoded_query, reduction_indices=[2])

        # Subtract max for numerical stability (softmax is shift invariant)
        attention_max = tf.reduce_max(attention, reduction_indices=[-1], keep_dims=True)
        attention = tf.nn.softmax(attention - attention_max)
        attention = tf.expand_dims(attention, 2)

        # Weight memories by attention vectors
        u = tf.reduce_sum(last_state * attention, reduction_indices=[1])

        # R acts as the decoder matrix to convert from internal state to the output vocabulary size
        R = tf.get_variable('R', [embedding_size, vocab_size])
        H = tf.get_variable('H', [embedding_size, embedding_size])

        q = tf.squeeze(encoded_query, squeeze_dims=[1])
        y = tf.matmul(activation(q + tf.matmul(u, H)), R)
        return y

def get_loss(output, labels, mode):
    if mode == tf.contrib.learn.ModeKeys.INFER:
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
