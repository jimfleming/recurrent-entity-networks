from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

class DynamicMemoryCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_blocks, num_units_per_block, activation=tf.identity):
        self._num_blocks = num_blocks # M
        self._num_units_per_block = num_units_per_block # d
        self._activation = activation # \phi

    @property
    def state_size(self):
        return self._num_blocks * self._num_units_per_block

    @property
    def output_size(self):
        return self._num_blocks * self._num_units_per_block

    def get_gate(self, inputs, state_j, key_j):
        # XXX: We assume the gate is a scalar.
        a = tf.reduce_sum(tf.mul(inputs, state_j), reduction_indices=[1])
        b = tf.reduce_sum(tf.mul(inputs, tf.expand_dims(key_j, 0)), reduction_indices=[1])
        return tf.sigmoid(a + b)

    def get_candidate(self, state_j, key_j, inputs, U, V, W):
        h_j_U = tf.matmul(state_j, U)
        s_t_W = tf.matmul(inputs, W)
        w_j_V = tf.matmul(tf.expand_dims(key_j, -1), V, transpose_a=True)
        return self._activation(h_j_U + w_j_V + s_t_W)

    def __call__(self, inputs, state, scope=None):
        state = tf.split(1, self._num_blocks, state)

        # split into blocks (U, V, W are shared)
        with tf.variable_scope(scope or type(self).__name__):
            next_states = []
            for j, state_j in enumerate(state): # Hidden State (j)
                key_j = tf.get_variable('key_{}'.format(j),
                    shape=[self._num_units_per_block],
                    initializer=tf.contrib.layers.variance_scaling_initializer()) # Key Vector (j)

                reuse = False if j == 0 else True
                with tf.variable_scope('Gate', reuse=reuse):
                    gate_j = self.get_gate(inputs, state_j, key_j) # 2) Gate

                with tf.variable_scope('Candidate', reuse=reuse):
                    U = tf.get_variable('U',
                        shape=[self._num_units_per_block, self._num_units_per_block],
                        initializer=tf.contrib.layers.variance_scaling_initializer())
                    V = tf.get_variable('V',
                        shape=[self._num_units_per_block, self._num_units_per_block],
                        initializer=tf.contrib.layers.variance_scaling_initializer())
                    W = tf.get_variable('W',
                        shape=[self._num_units_per_block, self._num_units_per_block],
                        initializer=tf.contrib.layers.variance_scaling_initializer())
                    candidate = self.get_candidate(state_j, key_j, inputs, U, V, W) # 3) Candidate

                state_j_next = state_j + tf.expand_dims(gate_j, -1) * candidate # 4) Update
                state_j_next = tf.nn.l2_normalize(state_j_next, -1) # 5) Forget (via normalization)

                next_states.append(state_j_next)

        state_next = tf.concat(1, next_states)
        return state_next, state_next
