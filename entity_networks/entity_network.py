from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from entity_networks.activations import prelu
from entity_networks.dynamic_memory_cell import DynamicMemoryCell

class EntityNetworkModel(object):

    def __init__(self, story, query, answer, batch_size, is_training=True):
        self.batch_size = batch_size

        self.max_sentence_length = 7
        self.max_story_length = 10
        self.max_query_length = 4
        self.vocab_size = 22

        self.num_blocks = 20
        self.embedding_size = 100

        embedding_params = tf.get_variable('embedding_params',
            shape=[self.vocab_size, self.embedding_size],
            initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))

        story_embedding = tf.nn.embedding_lookup(embedding_params, story)
        query_embedding = tf.nn.embedding_lookup(embedding_params, query)

        # Input Module
        encoded_story = self.encode_input(story_embedding, self.max_story_length, self.max_sentence_length, scope='StoryEncoding')

        # Dynamic Memory
        memory_cell = DynamicMemoryCell(self.num_blocks, self.embedding_size, activation=prelu)
        output, last_state = tf.nn.dynamic_rnn(memory_cell, encoded_story,
            initial_state=memory_cell.zero_state(batch_size, dtype=tf.float32),
            sequence_length=self.get_sequence_length(encoded_story))

        # Output Module
        self.output = self.get_output(last_state, query_embedding)

        # Loss
        with tf.variable_scope('Loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.output, answer)
            self.loss = tf.reduce_mean(cross_entropy)

        # Accuracy
        with tf.variable_scope('accuracy'):
            self.accuracy = tf.contrib.metrics.accuracy(tf.argmax(self.output, 1), answer)

        # Summaries
        tf.contrib.layers.summarize_tensor(self.loss)
        tf.contrib.layers.summarize_tensor(self.accuracy)

        # Optimization
        if is_training:
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.train_op = tf.contrib.layers.optimize_loss(
                self.loss,
                self.global_step,
                learning_rate=self.learning_rate,
                clip_gradients=40.0,
                optimizer='Adam')

    def get_sequence_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.cast(tf.reduce_sum(used, reduction_indices=1), tf.int32)
        return length

    def encode_input(self, embedding, num_sentences, max_sentence_length, scope=None):
        with tf.variable_scope(scope or 'Encoding'):
            embedding = tf.reshape(embedding,
                shape=[-1, num_sentences, max_sentence_length, self.embedding_size])
            mask = tf.get_variable('mask',
                shape=[max_sentence_length, self.embedding_size],
                initializer=tf.contrib.layers.variance_scaling_initializer())
            encoded_input = tf.reduce_sum(embedding * mask, reduction_indices=[2])
            return encoded_input

    def get_output(self, last_state, query_embedding):
        """
        Implementation of Section 2.3, Equation 6. This module is also described in more detail here:
        [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852).
        """
        query_encoding = self.encode_input(query_embedding, 1, self.max_query_length, scope='QueryEncoding')
        query_encoding = tf.squeeze(query_encoding, squeeze_dims=[1])

        last_states = tf.split(1, self.num_blocks, last_state)

        # Use query to attend over memories (hidden states)
        p = []
        for h_j in last_states:
            p_j = tf.reduce_sum(query_encoding * h_j, 1, keep_dims=True)
            p.append(p_j)
        p = tf.concat(1, p)
        p = tf.nn.softmax(p) # Attention
        p = tf.split(1, self.num_blocks, p)

        # Weight memories by attention vectors
        u = []
        for h_j, p_j in zip(last_states, p):
            u_j = tf.expand_dims(h_j * p_j, -1)
            u.append(u_j)
        u = tf.concat(2, u)
        u = tf.reduce_sum(u, reduction_indices=[2])

        # R acts as the decoder matrix to convert from internal state to the output vocabulary size.
        R = tf.get_variable('R',
            shape=[self.embedding_size, self.vocab_size],
            initializer=tf.contrib.layers.variance_scaling_initializer())
        H = tf.get_variable('H',
            shape=[self.embedding_size, self.embedding_size],
            initializer=tf.contrib.layers.variance_scaling_initializer())

        y = tf.matmul(prelu(query_encoding + tf.matmul(u, H)), R)
        return y

    @property
    def num_parameters(self):
        return np.sum([np.prod(tvar.get_shape().as_list()) for tvar in tf.trainable_variables()])
