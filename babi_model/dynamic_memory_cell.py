from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

class DynamicMemoryCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_blocks, num_units_per_block, key_size, activation=tf.nn.relu):
        self._num_blocks = num_blocks # M
        self._num_units_per_block = num_units_per_block # d
        self._key_size = key_size # len(w_j)
        self._activation = activation # \phi

    @property
    def state_size(self):
        return self._num_blocks * self._num_units_per_block

    @property
    def output_size(self):
        return self._num_blocks * self._num_units_per_block

    def __call__(self, inputs, state, scope=None):
        state = tf.split(1, self._num_blocks, state)
        input_size = inputs.get_shape().with_rank(2)[1] # 32 x 20

        # split into blocks (U, V, W are shared)
        with tf.variable_scope(scope or type(self).__name__):
            state_next = []
            for j in range(self._num_blocks):
                key_j = tf.get_variable('key_{}'.format(j), shape=[input_size, self._num_units_per_block]) # key vector (j): 20 x 100
                state_j = state[j] # hidden state (j): ? x 100

                with tf.variable_scope('Gate'):
                    gate_j = tf.sigmoid(tf.matmul(inputs, state_j) + tf.matmul(inputs, key_j)) # gate (j): ? x 100

                reuse = False if j == 0 else True
                with tf.variable_scope('Candidate', reuse=reuse):
                    W = tf.get_variable('W', [input_size, self._num_units_per_block]) # 20 x 100
                    candidate = self._activation(tf.matmul(inputs, W)) # ? x 100

                state_j_next = state_j + gate_j * candidate
                state_j_next = tf.nn.l2_normalize(state_j_next, -1) # forget by normalization

                state_next.append(state_j_next)

        state_next = tf.concat(1, state_next)
        return state_next, state_next
