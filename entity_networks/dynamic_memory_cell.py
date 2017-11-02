"Define a dynamic memory cell."
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

class DynamicMemoryCell(tf.contrib.rnn.RNNCell):
    """
    Implementation of a dynamic memory cell as a gated recurrent network.
    The cell's hidden state is divided into blocks and each block's weights are tied.
    """

    def __init__(self,
                 num_blocks,
                 num_units_per_block,
                 keys,
                 initializer=None,
                 recurrent_initializer=None,
                 activation=tf.nn.relu):
        self._num_blocks = num_blocks # M
        self._num_units_per_block = num_units_per_block # d
        self._keys = keys
        self._activation = activation # \phi
        self._initializer = initializer
        self._recurrent_initializer = recurrent_initializer

    @property
    def state_size(self):
        "Return the total state size of the cell, across all blocks."
        return self._num_blocks * self._num_units_per_block

    @property
    def output_size(self):
        "Return the total output size of the cell, across all blocks."
        return self._num_blocks * self._num_units_per_block

    def zero_state(self, batch_size, dtype):
        "Initialize the memory to the key values."
        zero_state = tf.concat([tf.expand_dims(key, axis=0) for key in self._keys], axis=1)
        zero_state_batch = tf.tile(zero_state, [batch_size, 1])
        return zero_state_batch

    def get_gate(self, state_j, key_j, inputs):
        """
        Implements the gate (scalar for each block). Equation 2:

        g_j <- \sigma(s_t^T h_j + s_t^T w_j)
        """
        a = tf.reduce_sum(inputs * state_j, axis=1)
        b = tf.reduce_sum(inputs * key_j, axis=1)
        return tf.sigmoid(a + b)

    def get_candidate(self, state_j, key_j, inputs, U, V, W, U_bias):
        """
        Represents the new memory candidate that will be weighted by the
        gate value and combined with the existing memory. Equation 3:

        h_j^~ <- \phi(U h_j + V w_j + W s_t)
        """
        key_V = tf.matmul(key_j, V)
        state_U = tf.matmul(state_j, U) + U_bias
        inputs_W = tf.matmul(inputs, W)
        return self._activation(state_U + inputs_W + key_V)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, initializer=self._initializer):
            U = tf.get_variable('U', [self._num_units_per_block, self._num_units_per_block],
                                initializer=self._recurrent_initializer)
            V = tf.get_variable('V', [self._num_units_per_block, self._num_units_per_block],
                                initializer=self._recurrent_initializer)
            W = tf.get_variable('W', [self._num_units_per_block, self._num_units_per_block],
                                initializer=self._recurrent_initializer)

            U_bias = tf.get_variable('U_bias', [self._num_units_per_block])

            # Split the hidden state into blocks (each U, V, W are shared across blocks).
            state = tf.split(state, self._num_blocks, axis=1)

            next_states = []
            for j, state_j in enumerate(state): # Hidden State (j)
                key_j = tf.expand_dims(self._keys[j], axis=0)
                gate_j = self.get_gate(state_j, key_j, inputs)
                candidate_j = self.get_candidate(state_j, key_j, inputs, U, V, W, U_bias)

                # Equation 4: h_j <- h_j + g_j * h_j^~
                # Perform an update of the hidden state (memory).
                state_j_next = state_j + tf.expand_dims(gate_j, -1) * candidate_j

                # Equation 5: h_j <- h_j / \norm{h_j}
                # Forget previous memories by normalization.
                state_j_next_norm = tf.norm(
                    tensor=state_j_next,
                    ord='euclidean',
                    axis=-1,
                    keep_dims=True)
                state_j_next_norm = tf.where(
                    tf.greater(state_j_next_norm, 0.0),
                    state_j_next_norm,
                    tf.ones_like(state_j_next_norm))
                state_j_next = state_j_next / state_j_next_norm

                next_states.append(state_j_next)
            state_next = tf.concat(next_states, axis=1)
        return state_next, state_next
