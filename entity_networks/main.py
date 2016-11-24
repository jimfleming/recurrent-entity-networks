from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import random
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from functools import partial

from entity_networks.model import model_fn
from entity_networks.dataset import Dataset
from entity_networks.monitors import ProgressMonitor

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size.')
tf.app.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs.')
tf.app.flags.DEFINE_string('model_dir', 'logs/{}'.format(int(time.time())), 'Log directory.')
tf.app.flags.DEFINE_string('dataset_path', 'datasets/processed/qa1_single-supporting-fact_10k.json', 'Dataset metadata path.')

def main(_):
    def input_fn(is_training, num_epochs):
        name = 'train' if is_training else 'test'
        shuffle = True if is_training else False
        dataset = Dataset(FLAGS.dataset_path, name,
            batch_size=FLAGS.batch_size,
            num_epochs=num_epochs,
            shuffle=shuffle)
        features = {
            'story': dataset.story_batch,
            'query': dataset.query_batch,
        }
        labels = dataset.answer_batch
        return features, labels

    train_input_fn = partial(input_fn, is_training=True, num_epochs=None)
    eval_input_fn = partial(input_fn, is_training=False, num_epochs=1)

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
        train_monitors=None)

    experiment.train_and_evaluate()

if __name__ == '__main__':
    tf.app.run()
