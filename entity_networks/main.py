from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

SEED = 67

import time
import random; random.seed(SEED)
import numpy as np; np.random.seed(SEED)
import tensorflow as tf; tf.set_random_seed(SEED)

from entity_networks.entity_network import EntityNetworkModel
from entity_networks.trainer import Trainer
from entity_networks.data import input_pipeline

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size.')
tf.app.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs.')
tf.app.flags.DEFINE_string('logdir', 'logs/{}'.format(int(time.time())), 'Log directory.')

def main(_):
    with tf.device('/cpu:0'):
        story_train, query_train, answer_train = input_pipeline(
            filenames=['datasets/processed/qa1_single-supporting-fact_train.tfrecords'],
            batch_size=FLAGS.batch_size,
            num_epochs=FLAGS.num_epochs,
            shuffle=True)
        story_test, query_test, answer_test = input_pipeline(
            filenames=['datasets/processed/qa1_single-supporting-fact_test.tfrecords'],
            batch_size=FLAGS.batch_size,
            shuffle=False)

    with tf.variable_scope('model'):
        model_train = EntityNetworkModel(story_train, query_train, answer_train,
            batch_size=FLAGS.batch_size,
            is_training=True)

    with tf.variable_scope('model', reuse=True):
        model_test = EntityNetworkModel(story_test, query_test, answer_test,
            batch_size=FLAGS.batch_size,
            is_training=False)

    supervisor = tf.train.Supervisor(
        global_step=model_train.global_step,
        logdir=FLAGS.logdir,
        save_summaries_secs=1)

    with supervisor.managed_session() as sess:
        print('Training model with {} parameters'.format(model_train.num_parameters))
        trainer = Trainer(supervisor, sess, model_train, model_test)
        trainer.train()

if __name__ == '__main__':
    tf.app.run()
