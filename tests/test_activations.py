from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from entity_networks.activations import prelu

class PReLUTest(tf.test.TestCase):

    def test_identity(self):
        features = tf.constant([1, 0, -1], dtype=tf.float32)
        activation = prelu(features, tf.constant_initializer(1.0))
        with self.test_session():
            init = tf.initialize_all_variables()
            init.run()
            self.assertAllEqual(activation.eval(), [1, 0, -1])

    def test_relu(self):
        features = tf.constant([1, 0, -1], dtype=tf.float32)
        activation = prelu(features, tf.constant_initializer(0.0))
        with self.test_session():
            init = tf.initialize_all_variables()
            init.run()
            self.assertAllEqual(activation.eval(), [1, 0, 0])

if __name__ == '__main__':
    tf.test.main()
