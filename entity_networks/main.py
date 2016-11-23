from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import random
import numpy as np
import tensorflow as tf

from entity_networks.model import Model
from entity_networks.dataset import Dataset
from entity_networks.monitors import ProgressMonitor

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size.')
tf.app.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs.')
tf.app.flags.DEFINE_string('logdir', 'logs/{}'.format(int(time.time())), 'Log directory.')

def main(_):
    with tf.device('/cpu:0'):
        dataset_train = Dataset(
            filename='datasets/processed/qa1_single-supporting-fact_train.tfrecords',
            batch_size=FLAGS.batch_size,
            shuffle=True)
        dataset_test = Dataset(
            filename='datasets/processed/qa1_single-supporting-fact_test.tfrecords',
            batch_size=FLAGS.batch_size,
            shuffle=False)

    with tf.variable_scope('Model'):
        model_train = Model(dataset_train, is_training=True)

    with tf.variable_scope('Model', reuse=True):
        model_test = Model(dataset_test, is_training=False)

    print('Model has {} parameters'.format(model_train.num_parameters))
    print('Saving logs to {}'.format(FLAGS.logdir))

    monitors = [
        tf.contrib.learn.monitors.SummarySaver(
            summary_op=tf.merge_all_summaries(),
            save_steps=1,
            output_dir=FLAGS.logdir),
        ProgressMonitor(tensor_names={
            'Loss (Train)': model_train.loss.name,
            'Loss (Test)': model_test.loss.name,
            'LR': model_train.learning_rate.name,
        }, decay_rate=0.99),
    ]

    tf.contrib.learn.train(
        graph=tf.get_default_graph(),
        output_dir=FLAGS.logdir,
        train_op=model_train.train_op,
        loss_op=model_train.loss,
        supervisor_save_summaries_steps=1,
        monitors=monitors,
        log_every_steps=1,
        max_steps=FLAGS.num_epochs*dataset_train.num_batches)

if __name__ == '__main__':
    tf.app.run()
