from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from babi_model.activations import prelu

from keras.metrics import categorical_accuracy
from babi_model.dynamic_memory_cell import DynamicMemoryCell

class Model(object):

    def __init__(self, story, query, answer, story_length, query_length, batch_size, is_training=True):
        self.vocab_size = 22
        self.max_story_length = 68
        self.max_query_length = 4

        self.num_blocks = 20
        self.num_units_per_block = 100

        embedding_params = tf.get_variable('embedding_params',
            shape=[self.vocab_size, self.num_units_per_block],
            initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

        story_embedding = tf.nn.embedding_lookup(embedding_params, story)
        query_embedding = tf.nn.embedding_lookup(embedding_params, query)
        answer_hot = tf.one_hot(answer, self.vocab_size)

        # Input Module
        encoding_cell = tf.nn.rnn_cell.GRUCell(self.num_units_per_block, activation=prelu)

        # Dynamic Memory
        memory_cell = DynamicMemoryCell(self.num_blocks, self.num_units_per_block, activation=prelu)

        cell = tf.nn.rnn_cell.MultiRNNCell([encoding_cell, memory_cell])
        output, last_state = tf.nn.dynamic_rnn(cell, story_embedding,
            initial_state=cell.zero_state(batch_size, dtype=tf.float32),
            sequence_length=story_length)

        # Output Module
        self.output = self.get_output(last_state[1], query_embedding)

        # Loss
        with tf.variable_scope('Loss'):
            # XXX: We assume cross-entropy loss, even though the logits are never scaled.
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, answer_hot)
            self.loss = tf.reduce_mean(cross_entropy)

        # Accuracy
        with tf.variable_scope('accuracy'):
            self.accuracy = categorical_accuracy(answer_hot, self.output)

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

    def get_output(self, last_state, query_embedding):
        """
        Implementation of Section 2.3, Equation 6. This module is also described here:
        [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852).
        """
        # XXX: We assume the query is convert into a bag-of-words representation.
        q = tf.reduce_sum(query_embedding, reduction_indices=[1])
        last_states = tf.split(1, self.num_blocks, last_state)

        u_j = []
        for h_j in last_states:
            # XXX: We assume the matrix multiplication is correct as the inner product would be scalar.
            p_j_inner = tf.batch_matmul(tf.expand_dims(q, -1), tf.expand_dims(h_j, 1))
            p_j = tf.nn.softmax(p_j_inner) # Attention
            p_j = tf.batch_matmul(p_j, tf.expand_dims(h_j, -1))
            u_j.append(p_j)
        u_j = tf.concat(2, u_j)
        u = tf.reduce_sum(u_j, reduction_indices=[-1]) # Response Vector

        R = tf.get_variable('R',
            shape=[self.num_units_per_block, self.vocab_size],
            initializer=tf.contrib.layers.variance_scaling_initializer()) # Decoder Matrix
        H = tf.get_variable('H',
            shape=[self.num_units_per_block, self.num_units_per_block],
            initializer=tf.contrib.layers.variance_scaling_initializer())

        # XXX: We assume \phi in Equation 6 is the same \phi in Equation 3, which is PReLU for bAbI.
        y = tf.matmul(prelu(q + tf.matmul(u, H)), R)
        return y

    @property
    def num_parameters(self):
        return np.sum([np.prod(tvar.get_shape().as_list()) for tvar in tf.trainable_variables()])
