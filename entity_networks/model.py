"""
Define the recurrent entity network model.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from functools import partial

import tensorflow as tf

from entity_networks.dynamic_memory_cell import DynamicMemoryCell
from entity_networks.model_ops import cyclic_learning_rate, \
                                      get_sequence_length, \
                                      count_parameters, \
                                      prelu

OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "loss",
    "gradients",
    "gradient_norm",
]

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

def get_output_module(
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
    outputs = None
    return outputs

def get_outputs(inputs, params):
    "Return the outputs from the model which will be used in the loss function."
    embedding_size = params['embedding_size']
    num_blocks = params['num_blocks']
    vocab_size = params['vocab_size']

    story = inputs['story']
    query = inputs['query']

    batch_size = tf.shape(story)[0]

    normal_initializer = tf.random_normal_initializer(stddev=0.1)
    ones_initializer = tf.constant_initializer(1.0)

    # Extend the vocab to include keys for the dynamic memory cell,
    # allowing the initialization of the memory to be learned.
    vocab_size = vocab_size + num_blocks

    with tf.variable_scope('EntityNetwork', initializer=normal_initializer):
        # PReLU activations have their alpha parameters initialized to 1
        # so they may be identity before training.
        alpha = tf.get_variable(
            name='alpha',
            shape=embedding_size,
            initializer=ones_initializer)
        activation = partial(prelu, alpha=alpha)

        # Embeddings
        embedding_params = tf.get_variable(
            name='embedding_params',
            shape=[vocab_size, embedding_size])

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
        keys = tf.nn.embedding_lookup(embedding_params_masked, keys)
        keys = tf.split(keys, num_blocks, axis=0)
        keys = [tf.squeeze(key, axis=0) for key in keys]

        cell = DynamicMemoryCell(
            num_blocks=num_blocks,
            num_units_per_block=embedding_size,
            keys=keys,
            initializer=normal_initializer,
            recurrent_initializer=normal_initializer,
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
        outputs = get_output_module(
            last_state=last_state,
            encoded_query=encoded_query,
            num_blocks=num_blocks,
            vocab_size=vocab_size,
            initializer=normal_initializer,
            activation=activation)

        parameters = count_parameters()
        print('Parameters: {}'.format(parameters))

        return outputs

def get_predictions(outputs):
    "Return the actual predictions for use with evaluation metrics or TF Serving."
    predictions = tf.argmax(outputs, axis=-1)
    return predictions

def get_loss(outputs, labels, mode):
    "Return the loss function which will be used with an optimizer."

    loss = None
    if mode == tf.contrib.learn.ModeKeys.INFER:
        return loss

    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=outputs,
        labels=labels)
    return loss

def get_train_op(loss, params, mode):
    "Return the trainining operation which will be used to train the model."

    train_op = None
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
        return train_op

    global_step = tf.contrib.framework.get_or_create_global_step()

    learning_rate = cyclic_learning_rate(
        learning_rate_min=params['learning_rate_min'],
        learning_rate_max=params['learning_rate_max'],
        step_size=params['learning_rate_step_size'],
        mode='triangular',
        global_step=global_step)
    tf.summary.scalar('learning_rate', learning_rate)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer='Adam',
        clip_gradients=params['clip_gradients'],
        gradient_noise_scale=params['gradient_noise_scale'],
        summaries=OPTIMIZER_SUMMARIES)

    return train_op

def model_fn(features, labels, mode, params):
    "Return ModelFnOps for use with Estimator."

    outputs = get_outputs(features, params)
    predictions = get_predictions(outputs)
    loss = get_loss(outputs, labels, mode)
    train_op = get_train_op(loss, params, mode)

    return tf.contrib.learn.ModelFnOps(
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        mode=mode)
