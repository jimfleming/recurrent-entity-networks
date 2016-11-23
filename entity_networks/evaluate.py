from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import random
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from entity_networks.model import Model
from entity_networks.dataset import Dataset
from entity_networks.monitors import ProgressMonitor

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size.')
tf.app.flags.DEFINE_string('checkpoint_path', None, 'Log directory.')
tf.app.flags.DEFINE_string('logdir', 'logs/{}'.format(int(time.time())), 'Log directory.')
tf.app.flags.DEFINE_string('dataset_path', 'datasets/processed/qa1_single-supporting-fact_10k.json', 'Dataset metadata path.')

def main(_):
    with tf.device('/cpu:0'):
        dataset = Dataset(FLAGS.dataset_path, 'test',
            batch_size=FLAGS.batch_size,
            num_epochs=1,
            shuffle=False)

    with tf.variable_scope('Model'):
        model = Model(dataset, is_training=False)

    eval_dict = {
        model.loss.name: model.loss,
        model.accuracy.name: model.accuracy,
    }

    tf.contrib.learn.evaluate(
        graph=tf.get_default_graph(),
        output_dir=FLAGS.logdir,
        checkpoint_path=FLAGS.checkpoint_path,
        eval_dict=eval_dict,
        log_every_steps=1)

if __name__ == '__main__':
    tf.app.run()
