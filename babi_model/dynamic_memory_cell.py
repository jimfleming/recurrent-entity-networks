from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def batch_matmul_v_v(a, b):
    """
    Compute the dot product of two batched vectors.

    a: [batch_size, d] or [d]
    b: [batch_size, d] or [d]
    returns: [batch_size]
    """
    return tf.reduce_sum(tf.mul(a, b), reduction_indices=[-1])

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
        inputs_state_j = batch_matmul_v_v(inputs, state_j)
        inputs_key_j = batch_matmul_v_v(inputs, key_j)
        gate_j = tf.sigmoid(inputs_state_j + inputs_key_j)
        gate_j = tf.expand_dims(gate_j, -1)
        return gate_j

    def get_candidate(self, inputs, W):
        return self._activation(tf.matmul(inputs, W))

    def __call__(self, inputs, state, scope=None):
        state = tf.split(1, self._num_blocks, state)
        input_size = inputs.get_shape().with_rank(2)[1]

        # split into blocks (U, V, W are shared)
        with tf.variable_scope(scope or type(self).__name__):
            next_states = []
            for j in range(self._num_blocks):
                state_j = state[j] # hidden state (j)

                with tf.variable_scope('Gate'):
                    key_j = tf.get_variable('key_{}'.format(j),
                        shape=[self._num_units_per_block]) # key vector (j)
                    gate_j = self.get_gate(inputs, state_j, key_j) # 2) gate

                reuse = False if j == 0 else True
                with tf.variable_scope('Candidate', reuse=reuse):
                    W = tf.get_variable('W',
                        shape=[input_size, self._num_units_per_block])
                    candidate = self.get_candidate(inputs, W) # 3) candidate

                state_j_next = state_j + gate_j * candidate # 4) update state
                state_j_next = tf.nn.l2_normalize(state_j_next, -1) # 5) forget by normalization

                next_states.append(state_j_next)

        state_next = tf.concat(1, next_states)
        return state_next, state_next
