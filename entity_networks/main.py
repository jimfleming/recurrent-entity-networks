from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

SEED = 67

import time
import random; random.seed(SEED)
import numpy as np; np.random.seed(SEED)
import tensorflow as tf; tf.set_random_seed(SEED)

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
            filenames=['datasets/processed/qa1_single-supporting-fact_train.tfrecords'],
            batch_size=FLAGS.batch_size,
            shuffle=True)
        dataset_test = Dataset(
            filenames=['datasets/processed/qa1_single-supporting-fact_test.tfrecords'],
            batch_size=FLAGS.batch_size,
            shuffle=False)

    with tf.variable_scope('Model'):
        model_train = Model(dataset_train, is_training=True)

    # TODO: use validation monitor
    with tf.variable_scope('Model', reuse=True):
        model_test = Model(dataset_test, is_training=False)

    print('Saving logs to {}'.format(FLAGS.logdir))
    print('Training model with {} parameters'.format(model_train.num_parameters))

    max_steps = FLAGS.num_epochs*dataset_train.num_batches
    monitors = [
        tf.contrib.learn.monitors.SummarySaver(
            summary_op=tf.merge_all_summaries(),
            save_steps=1,
            output_dir=FLAGS.logdir),
        ProgressMonitor(tensor_names={
            'Loss (Train)': model_train.loss.name,
            'Loss (Test)': model_test.loss.name,
            'LR': model_train.learning_rate.name,
        }, every_n_steps=1000, first_n_steps=0), # TODO: fix every N: logs too often
    ]

    tf.contrib.learn.train(
        graph=tf.get_default_graph(),
        output_dir=FLAGS.logdir,
        train_op=model_train.train_op,
        loss_op=model_train.loss,
        supervisor_save_summaries_steps=1,
        monitors=monitors,
        log_every_steps=1,
        max_steps=max_steps)

if __name__ == '__main__':
    tf.app.run()
