"Define the recurrent entity network model."
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from functools import partial

import numpy as np
import tensorflow as tf

from entity_networks.model_ops import get_sequence_length
from entity_networks.activation_ops import prelu
from entity_networks.dynamic_memory_cell import DynamicMemoryCell

def model_fn(features, labels, params, mode, scope=None):
    "Define the recurrent entity network model."
    embedding_size = params['embedding_size']
    num_blocks = params['num_blocks']
    vocab_size = params['vocab_size']
    debug = params['debug']

    story = features['story']
    query = features['query']

    batch_size = tf.shape(story)[0]

    normal_initializer = tf.random_normal_initializer(stddev=0.1)
    ones_initializer = tf.constant_initializer(1.0)

    # Extend the vocab to include keys for the dynamic memory cell,
    # allowing the initialization of the memory to be learned.
    vocab_size = vocab_size + num_blocks

    # PReLU activations have their alpha parameters initialized to 1
    # so they may be identity before training.
    alpha = tf.get_variable(
        name='alpha',
        shape=embedding_size,
        initializer=ones_initializer)
    activation = partial(prelu, alpha=alpha)

    with tf.variable_scope(scope, 'EntityNetwork', initializer=normal_initializer):
        # Embeddings
        embedding_params = tf.get_variable(
            name='embedding_params',
            shape=[vocab_size, embedding_size],
            initializer=normal_initializer)

        # The embedding mask forces the special "pad" embedding to zeros.
        embedding_mask = tf.constant(
            value=[0 if i == 0 else 1 for i in range(vocab_size)],
            shape=[vocab_size, 1],
            dtype=tf.float32)
        embedding_params_masked = embedding_params * embedding_mask

        story_embedding = tf.nn.embedding_lookup(embedding_params_masked, story)
        query_embedding = tf.nn.embedding_lookup(embedding_params_masked, query)

        # Input Module
        encoded_story = get_input_encoding(
            inputs=story_embedding,
            initializer=ones_initializer,
            scope='StoryEncoding')
        encoded_query = get_input_encoding(
            inputs=query_embedding,
            initializer=ones_initializer,
            scope='QueryEncoding')

        # Memory Module
        # We define the keys outside of the cell so they may be used for memory initialization.
        # Keys are initialized to a range outside of the main vocab.
        keys = [key for key in range(vocab_size - num_blocks, vocab_size)]
        keys = tf.constant(keys, shape=[num_blocks], dtype=tf.int32)
        keys = tf.nn.embedding_lookup(embedding_params_masked, keys)
        print('keys', keys)

        cell = DynamicMemoryCell(
            num_blocks=num_blocks,
            num_units_per_block=embedding_size,
            keys=keys,
            initializer=normal_initializer,
            activation=activation)

        # Recurrence
        initial_state = cell.zero_state(batch_size, tf.float32)
        sequence_length = get_sequence_length(encoded_story)
        _, last_state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=encoded_story,
            sequence_length=sequence_length,
            initial_state=initial_state)

        # Output Module
        outputs = get_outputs(
            last_state=last_state,
            encoded_query=encoded_query,
            num_blocks=num_blocks,
            vocab_size=vocab_size,
            initializer=normal_initializer,
            activation=activation)
        predictions = tf.argmax(outputs, 1)

        # Training
        loss = None
        if mode != tf.contrib.learn.ModeKeys.INFER:
            loss = tf.losses.sparse_softmax_cross_entropy(
                logits=outputs,
                labels=labels)

        train_op = None
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            train_op = get_train_op(loss, params)

        if debug:
            tf.summary.histogram(sequence_length, 'sequence_length')
            tf.summary.histogram(encoded_story, 'encoded_story')
            tf.summary.histogram(encoded_query, 'encoded_query')
            tf.summary.histogram(last_state, 'last_state')
            tf.summary.histogram(outputs, 'outputs')

            tf.add_check_numerics_ops()

        parameters = sum([np.prod(tvar.get_shape().as_list()) for tvar in tf.trainable_variables()])
        print('Parameters: {}'.format(parameters))

        return predictions, loss, train_op

def get_input_encoding(inputs, initializer=None, scope=None):
    """
    Implementation of the learned multiplicative mask from Section 2.1, Equation 1.
    This module is also described in [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852)
    as Position Encoding (PE). The mask allows the ordering of words in a sentence to affect the
    encoding.
    """
    with tf.variable_scope(scope, 'Encoding', initializer=initializer):
        _, _, max_sentence_length, embedding_size = inputs.get_shape().as_list()
        positional_mask = tf.get_variable(
            name='positional_mask',
            shape=[max_sentence_length, embedding_size])
        encoded_input = tf.reduce_sum(inputs * positional_mask, axis=2)
        return encoded_input

def get_outputs(
        last_state,
        encoded_query,
        num_blocks,
        vocab_size,
        activation=tf.nn.relu,
        initializer=None,
        scope=None):
    """
    Implementation of Section 2.3, Equation 6. This module is also described in more detail here:
    [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852).
    """
    with tf.variable_scope(scope, 'Output', initializer=initializer):
        last_state = tf.stack(tf.split(last_state, num_blocks, axis=1), axis=1)
        _, _, embedding_size = last_state.get_shape().as_list()

        # Use the encoded_query to attend over memories
        # (hidden states of dynamic last_state cell blocks)
        attention = tf.reduce_sum(last_state * encoded_query, axis=2)

        # Subtract max for numerical stability (softmax is shift invariant)
        attention_max = tf.reduce_max(attention, axis=-1, keep_dims=True)
        attention = tf.nn.softmax(attention - attention_max)
        attention = tf.expand_dims(attention, axis=2)

        # Weight memories by attention vectors
        u = tf.reduce_sum(last_state * attention, axis=1)

        # R acts as the decoder matrix to convert from internal state to the output vocabulary size
        R = tf.get_variable('R', [embedding_size, vocab_size])
        H = tf.get_variable('H', [embedding_size, embedding_size])

        q = tf.squeeze(encoded_query, axis=1)
        y = tf.matmul(activation(q + tf.matmul(u, H)), R)
        return y

def get_train_op(loss, params):
    "Get a training operation using Adam optimizer."
    global_step = tf.contrib.framework.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        learning_rate=params['learning_rate_init'],
        decay_steps=params['learning_rate_decay_steps'],
        decay_rate=params['learning_rate_decay_rate'],
        global_step=global_step,
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer='Adam',
        clip_gradients=params['clip_gradients'])

    return train_op
