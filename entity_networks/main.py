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
tf.app.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs.')
tf.app.flags.DEFINE_integer('seed', 67, 'Random seed.')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Base learning rate.')
tf.app.flags.DEFINE_float('clip_gradients', 40.0, 'Clip the global norm of the gradients to this value.')
tf.app.flags.DEFINE_string('model_dir', 'logs/', 'Model directory.')
tf.app.flags.DEFINE_string('dataset', 'datasets/processed/qa1_single-supporting-fact_10k.json', 'Dataset path.')

def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset = Dataset(FLAGS.dataset)

    train_input_fn = dataset.get_input_fn('train',
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs,
        shuffle=True)
    eval_input_fn = dataset.get_input_fn('test',
        batch_size=FLAGS.batch_size,
        num_epochs=1,
        shuffle=False)

    params = {
        'vocab_size': dataset.vocab_size,
        'embedding_size': FLAGS.embedding_size,
        'num_blocks': FLAGS.num_blocks,
        'learning_rate_init': FLAGS.learning_rate,
        'learning_rate_decay_steps': (10000 // FLAGS.batch_size) * 25,
        'learning_rate_decay_rate': 0.5,
        'clip_gradients': FLAGS.clip_gradients,
    }

    eval_metrics = {
        "accuracy": tf.contrib.learn.metric_spec.MetricSpec(tf.contrib.metrics.streaming_accuracy)
    }

    config = tf.contrib.learn.RunConfig(
        tf_random_seed=FLAGS.seed,
        save_summary_steps=120,
        save_checkpoints_secs=600,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=1,
        log_device_placement=True)

    dataset_name = os.path.splitext(os.path.basename(FLAGS.dataset))[0]
    timestamp = int(time.time())
    model_dir = os.path.join(FLAGS.model_dir, dataset_name, str(timestamp))
    estimator = tf.contrib.learn.Estimator(
        model_dir=model_dir,
        model_fn=model_fn,
        config=config,
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
