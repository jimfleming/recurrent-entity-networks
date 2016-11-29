from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time

import random
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

from entity_networks.model import model_fn
from entity_networks.dataset import Dataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size.')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Embedding size.')
tf.app.flags.DEFINE_integer('num_blocks', 20, 'Number of memory blocks.')
tf.app.flags.DEFINE_string('model_dir', 'logs/', 'Output directory.')
tf.app.flags.DEFINE_string('dataset', 'datasets/processed/qa1_single-supporting-fact_10k.json', 'Dataset path.')

def main(_):
    dataset = Dataset(FLAGS.dataset, FLAGS.batch_size)

    input_fn = dataset.get_input_fn('test',
        num_epochs=1,
        shuffle=False)

    params = {
        'vocab_size': dataset.vocab_size,
        'embedding_size': FLAGS.embedding_size,
        'num_blocks': FLAGS.num_blocks,
        'debug': False,
    }

    config = tf.contrib.learn.RunConfig(
        save_summary_steps=120,
        save_checkpoints_secs=600,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=1,
        log_device_placement=True)

    eval_metrics = {
        "accuracy": tf.contrib.learn.metric_spec.MetricSpec(tf.contrib.metrics.streaming_accuracy)
    }

    estimator = tf.contrib.learn.Estimator(
        model_dir=FLAGS.model_dir,
        model_fn=model_fn,
        config=config,
        params=params)

    experiment = tf.contrib.learn.Experiment(
        estimator,
        train_input_fn=None,
        eval_input_fn=input_fn,
        train_steps=None,
        eval_steps=None,
        eval_metrics=eval_metrics,
        train_monitors=None,
        local_eval_frequency=1)

    experiment.evaluate(delay_secs=10)

if __name__ == '__main__':
    tf.app.run()
