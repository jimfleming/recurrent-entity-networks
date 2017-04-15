"Main training module."
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time
import random
import numpy as np
import tensorflow as tf

from entity_networks.model import model_fn
from entity_networks.dataset import Dataset

TIMESTAMP = int(time.time())
RUN_NAME = os.environ.get('RUN_NAME', str(TIMESTAMP))
SHARED_DIR = os.environ.get('SHARED_DIR', None)

if SHARED_DIR is not None:
    DATA_DIR = os.path.join(SHARED_DIR, 'data/babi/records/')
    MODEL_DIR = os.path.join(SHARED_DIR, 'runs', RUN_NAME)
else:
    DATA_DIR = 'data/records/'
    MODEL_DIR = os.path.join('logs', RUN_NAME)

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size.')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Embedding size.')
tf.app.flags.DEFINE_integer('num_blocks', 20, 'Number of memory blocks.')
tf.app.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs.')
tf.app.flags.DEFINE_integer('seed', 67, 'Random seed.')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Base learning rate.')
tf.app.flags.DEFINE_float('clip_gradients', 40.0, 'Clip gradient global norm to this value.')
tf.app.flags.DEFINE_boolean('debug', False, 'Enable more summaries and numerical checks.')
tf.app.flags.DEFINE_string('dataset', 'qa1_single-supporting-fact_10k.json', 'Dataset path.')

def main(_):
    "Main training entrypoint."

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset_path = os.path.join(DATA_DIR, FLAGS.dataset)
    dataset = Dataset(dataset_path, FLAGS.batch_size)

    train_input_fn = dataset.get_input_fn(
        dataset_name='train',
        num_epochs=FLAGS.num_epochs,
        shuffle=True)
    eval_input_fn = dataset.get_input_fn(
        dataset_name='test',
        num_epochs=1,
        shuffle=False)

    params = {
        'vocab_size': dataset.vocab_size,
        'embedding_size': FLAGS.embedding_size,
        'num_blocks': FLAGS.num_blocks,
        'learning_rate_init': FLAGS.learning_rate,
        'learning_rate_decay_steps': dataset.steps_per_epoch * 25,
        'learning_rate_decay_rate': 0.5,
        'clip_gradients': FLAGS.clip_gradients,
        'debug': FLAGS.debug,
    }

    eval_metrics = {
        'accuracy': tf.contrib.learn.MetricSpec(tf.contrib.metrics.streaming_accuracy)
    }

    config = tf.contrib.learn.RunConfig(
        tf_random_seed=FLAGS.seed,
        save_summary_steps=100,
        save_checkpoints_steps=dataset.steps_per_epoch,
        save_checkpoints_secs=None,
        keep_checkpoint_max=5,
        log_device_placement=False,
        gpu_memory_fraction=0.8)

    estimator = tf.contrib.learn.Estimator(
        model_dir=MODEL_DIR,
        model_fn=model_fn,
        config=config,
        params=params)

    export_strategy = tf.contrib.learn.make_export_strategy(
        serving_input_fn=dataset.get_serving_input_fn(),
        default_output_alternative_key=None,
        assets_extra=None,
        as_text=False,
        exports_to_keep=5)

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=None,
        eval_steps=None,
        eval_metrics=eval_metrics,
        train_monitors=None,
        local_eval_frequency=1,
        export_strategies=[export_strategy])

    experiment.train_and_evaluate()

if __name__ == '__main__':
    tf.app.run()
