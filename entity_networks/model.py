from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from functools import partial

from entity_networks.activations import prelu
from entity_networks.dynamic_memory_cell import DynamicMemoryCell

def count_parameters_in_scope(scope=None):
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    return np.sum([np.prod(var.get_shape().as_list()) for var in variables])

class Model(object):

    def __init__(self, dataset, is_training=True):
        self.dataset = dataset
        self.is_training = is_training

        self.num_blocks = 20
        self.embedding_size = 100

        self.prelu_ones = partial(prelu, initializer=tf.constant_initializer(1.0))

        embedding_params = tf.get_variable('embedding_params',
            shape=[dataset.vocab_size, self.embedding_size],
            initializer=tf.random_normal_initializer(stddev=0.1))

        story_embedding = tf.nn.embedding_lookup(embedding_params, dataset.story_batch)
        query_embedding = tf.nn.embedding_lookup(embedding_params, dataset.query_batch)

        # Mask embeddings
        # TODO: check embeddings are correctly masked
        story_mask = self.get_padding_mask(dataset.story_batch)
        query_mask = self.get_padding_mask(dataset.query_batch)

        # Input Module
        encoded_story = self.encode_input(story_embedding * story_mask, scope='StoryEncoding')
        encoded_query = self.encode_input(query_embedding * query_mask, scope='QueryEncoding')

        # Dynamic Memory
        sequence_length = self.get_sequence_length(encoded_story)
        cell = DynamicMemoryCell(self.num_blocks, self.embedding_size,
            activation=self.prelu_ones)
        initial_state = cell.zero_state(dataset.batch_size, dtype=tf.float32)
        _, last_state = tf.nn.dynamic_rnn(cell, encoded_story,
            initial_state=initial_state,
            sequence_length=sequence_length)

        # Output Module
        self.output = self.get_output(last_state, encoded_query)

        # Loss
        with tf.variable_scope('Loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.output, dataset.answer_batch)
            self.loss = tf.reduce_mean(cross_entropy)

        # Accuracy
        with tf.variable_scope('Accuracy'):
            self.accuracy = tf.contrib.metrics.accuracy(tf.argmax(self.output, 1), dataset.answer_batch)

        # Summaries
        tf.contrib.layers.summarize_tensor(self.loss)
        tf.contrib.layers.summarize_tensor(self.accuracy)

        # Optimization
        if is_training:
            self.global_step = tf.contrib.framework.get_or_create_global_step()

            with tf.variable_scope('LearningRate'):
                num_steps_per_decay = dataset.num_batches * 25
                self.learning_rate = 1e-2 / 2**(tf.to_float(self.global_step) // num_steps_per_decay)

            tf.contrib.layers.summarize_tensor(self.learning_rate)
            tf.contrib.layers.summarize_tensor(embedding_params)
            tf.contrib.layers.summarize_variables(name_filter='alpha')
            tf.contrib.layers.summarize_activations(name_filter='PReLU')

            self.train_op = tf.contrib.layers.optimize_loss(
                self.loss,
                self.global_step,
                learning_rate=self.learning_rate,
                clip_gradients=40.0,
                optimizer='Adam')

    def get_sequence_length(self, sequence):
        """
        This is a hacky way of determining the actual length of a sequence that has been padded with zeros.
        """
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=[-1]))
        length = tf.cast(tf.reduce_sum(used, reduction_indices=[-1]), tf.int32)
        return length

    def get_padding_mask(self, sequence):
        """
        This is a hacky way of masking the padded sentence embeddings.
        """
        sequence = tf.reduce_sum(sequence, reduction_indices=[-1], keep_dims=True)
        mask = tf.to_float(tf.greater(sequence, 0))
        return tf.expand_dims(mask, -1)

    def encode_input(self, embedding, scope=None):
        """
        Implementation of the learned multiplicative mask from Section 2.1, Equation 1. This module is also described
        in [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852) as Position Encoding (PE). The mask allows
        the ordering of words in a sentence to affect the encoding.
        """
        with tf.variable_scope(scope, 'Encoding'):
            _, _, max_sentence_length, _ = embedding.get_shape().as_list()
            positional_mask = tf.get_variable('positional_mask',
                shape=[max_sentence_length, 1],
                initializer=tf.random_normal_initializer(stddev=0.1))
            encoded_input = tf.reduce_sum(embedding * positional_mask, reduction_indices=[2])
            return encoded_input

    def get_output(self, last_state, encoded_query, scope=None):
        """
        Implementation of Section 2.3, Equation 6. This module is also described in more detail here:
        [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852).
        """
        with tf.variable_scope(scope, 'Output'):
            last_state = tf.pack(tf.split(1, self.num_blocks, last_state), axis=1)

            # Use query to attend over memories (hidden states of dynamic memory cell blocks)
            p = tf.reduce_sum(last_state * encoded_query, reduction_indices=[2])
            p = tf.nn.softmax(p)

            # Weight memories by attention vectors
            u = tf.reduce_sum(last_state * tf.expand_dims(p, 2), reduction_indices=[1])

            # R acts as the decoder matrix to convert from internal state to the output vocabulary size.
            R = tf.get_variable('R',
                shape=[self.embedding_size, self.dataset.vocab_size],
                initializer=tf.random_normal_initializer(stddev=0.1))
            H = tf.get_variable('H',
                shape=[self.embedding_size, self.embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.1))

            y = tf.matmul(self.prelu_ones(tf.squeeze(encoded_query) + tf.matmul(u, H)), R)
            return y

    @property
    def num_parameters(self):
        return count_parameters_in_scope()
