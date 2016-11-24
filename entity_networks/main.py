from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import random
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from entity_networks.model import model_fn
from entity_networks.dataset import Dataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size.')
tf.app.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs.')
tf.app.flags.DEFINE_string('model_dir', 'logs/{}'.format(int(time.time())), 'Log directory.')
tf.app.flags.DEFINE_string('dataset_dir', 'datasets/processed/', 'Dataset directory.')
tf.app.flags.DEFINE_string('dataset', 'qa1_single-supporting-fact_10k.json', 'Dataset directory.')

def main(_):
    dataset = Dataset(FLAGS.dataset_dir, FLAGS.dataset)

    train_input_fn = dataset.get_input_fn('train',
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs,
        shuffle=True)
    eval_input_fn = dataset.get_input_fn('test',
        batch_size=FLAGS.batch_size,
        num_epochs=1,
        shuffle=False)

    params = {
        'embedding_size': 100,
        'num_blocks': 20,
        'vocab_size': 22,
        'learning_rate_init': 1e-2,
        'learning_rate_decay_steps': (10000 // FLAGS.batch_size) * 25,
        'learning_rate_decay_rate': 0.5,
        'clip_gradients': 40.0,
    }

    eval_metrics = {"accuracy": tf.contrib.metrics.streaming_accuracy}

    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        params=params)

    experiment = tf.contrib.learn.Experiment(
        estimator,
        train_input_fn,
        eval_input_fn,
        train_steps=None,
        eval_steps=None,
        eval_metrics=eval_metrics,
        train_monitors=None,
        local_eval_frequency=1)

    experiment.train_and_evaluate()

if __name__ == '__main__':
    tf.app.run()
