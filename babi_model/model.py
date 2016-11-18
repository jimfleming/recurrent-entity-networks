from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from keras.metrics import categorical_accuracy
from babi_model.dynamic_memory_cell import DynamicMemoryCell

class Model(object):

    def __init__(self, story, query, answer, story_length, query_length, is_training=True):
        self.batch_size = 32
        self.vocab_size = 22
        self.max_story_length = 66
        self.max_query_length = 4

        self.num_blocks = 5
        self.num_units_per_block = 100
        self.embedding_size = 20

        self.window_size = 6
        self.num_windows = self.max_story_length // self.window_size

        story = tf.reshape(story, [self.batch_size, self.max_story_length])
        query = tf.reshape(query, [self.batch_size, self.max_query_length])

        embedding_params = tf.get_variable('embedding_params', [self.vocab_size, self.embedding_size],
            initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

        story_embedding = tf.nn.embedding_lookup(embedding_params, story) # [batch_size, max_story_length, embedding_size]
        query_embedding = tf.nn.embedding_lookup(embedding_params, query) # [batch_size, max_query_length, embedding_size]

        # Input Encoder
        # The encoding layer summarizes an element of the input sequence with a vector of fixed length.
        # TODO: Try replacing this with a GRU + MultiRNNCell
        state = self.encode_input(story_embedding)

        # Dynamic Memory
        cell = DynamicMemoryCell(self.num_blocks, self.num_units_per_block, self.embedding_size, activation=tf.identity)
        _, last_state = tf.nn.dynamic_rnn(cell, state, sequence_length=story_length // self.window_size + 1, dtype=tf.float32)

        # Output Module
        self.output = self.get_output(last_state, query_embedding)
        answer_hot = tf.one_hot(answer, self.vocab_size)

        # Loss
        with tf.variable_scope('Loss'):
            cross_entropy = -tf.reduce_sum(answer_hot * tf.log(self.output + 1e-12))
            self.loss = tf.reduce_mean(cross_entropy)

        # Accuracy
        with tf.variable_scope('accuracy'):
            self.accuracy = categorical_accuracy(answer, self.output)

        # Summaries
        tf.contrib.layers.summarize_tensor(self.loss)
        tf.contrib.layers.summarize_tensor(self.accuracy)

        # Optimization
        if is_training:
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.train_op = tf.contrib.layers.optimize_loss(
                self.loss,
                self.global_step,
                learning_rate=1e-3,
                optimizer='Adam')

    def encode_input(self, story_embedding):
        story_embedding_window = tf.reshape(story_embedding, [
            self.batch_size,
            self.num_windows,
            self.window_size,
            self.embedding_size,
        ])
        multiplicative_mask = tf.get_variable('multiplicative_mask', [self.window_size, self.embedding_size],
            initializer=tf.contrib.layers.variance_scaling_initializer())
        state = tf.reduce_sum(story_embedding_window * multiplicative_mask, reduction_indices=[2])
        return state

    def get_output(self, last_state, query_embedding):
        q = tf.reduce_sum(query_embedding, reduction_indices=[1]) # BoW query: 32 x 20
        last_states = tf.split(1, self.num_blocks, last_state) # [? x 20, ..., ? x 20]

        u_j = []
        for j in range(self.num_blocks):
            p_j = tf.nn.softmax(tf.matmul(q, last_states[j]), dim=-1) # 32 x 20
            p_j = tf.matmul(p_j, last_states[j])
            p_j = tf.expand_dims(p_j, 2)
            u_j.append(p_j)
        u_j = tf.concat(2, u_j)
        u = tf.reduce_sum(u_j, reduction_indices=[2]) # response vector

        R = tf.get_variable('R', [self.embedding_size, self.vocab_size]) # decoder matrix
        H = tf.get_variable('H', [self.num_units_per_block, self.embedding_size]) # 20 x 20

        y = tf.matmul(tf.nn.relu(tf.matmul(u, H) + q), R)
        return y

    @property
    def num_parameters(self):
        return np.sum([np.prod(tvar.get_shape().as_list()) for tvar in tf.trainable_variables()])
