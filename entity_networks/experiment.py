from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import tensorflow as tf

from entity_networks.hooks import EarlyStoppingHook
from entity_networks.inputs import generate_input_fn
from entity_networks.serving import generate_serving_input_fn
from entity_networks.model import model_fn

BATCH_SIZE = 32
NUM_BLOCKS = 20
EMBEDDING_SIZE = 100
CLIP_GRADIENTS = 40.0

def generate_experiment_fn(data_dir, dataset_id, num_epochs,
                           learning_rate_min, learning_rate_max,
                           learning_rate_step_size, gradient_noise_scale):
    "Return _experiment_fn for use with learn_runner."
    def _experiment_fn(output_dir):
        metadata_path = os.path.join(data_dir, '{}_10k.json'.format(dataset_id))
        with tf.gfile.Open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)

        train_filename = os.path.join(data_dir, '{}_10k_{}.tfrecords'.format(dataset_id, 'train'))
        eval_filename = os.path.join(data_dir, '{}_10k_{}.tfrecords'.format(dataset_id, 'test'))

        train_input_fn = generate_input_fn(
            filename=train_filename,
            metadata=metadata,
            batch_size=BATCH_SIZE,
            num_epochs=num_epochs,
            shuffle=True)

        eval_input_fn = generate_input_fn(
            filename=eval_filename,
            metadata=metadata,
            batch_size=BATCH_SIZE,
            num_epochs=1,
            shuffle=False)

        vocab_size = metadata['vocab_size']
        task_size = metadata['task_size']
        train_steps_per_epoch = task_size // BATCH_SIZE

        run_config = tf.contrib.learn.RunConfig(
            save_summary_steps=train_steps_per_epoch,
            save_checkpoints_steps=5 * train_steps_per_epoch,
            save_checkpoints_secs=None)

        params = {
            'vocab_size': vocab_size,
            'embedding_size': EMBEDDING_SIZE,
            'num_blocks': NUM_BLOCKS,
            'learning_rate_min': learning_rate_min,
            'learning_rate_max': learning_rate_max,
            'learning_rate_step_size': learning_rate_step_size * train_steps_per_epoch,
            'clip_gradients': CLIP_GRADIENTS,
            'gradient_noise_scale': gradient_noise_scale,
        }

        estimator = tf.contrib.learn.Estimator(
            model_dir=output_dir,
            model_fn=model_fn,
            config=run_config,
            params=params)

        eval_metrics = {
            'accuracy': tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy)
        }

        train_monitors = [
            EarlyStoppingHook(
                input_fn=eval_input_fn,
                estimator=estimator,
                metrics=eval_metrics,
                metric_name='accuracy',
                every_steps=5 * train_steps_per_epoch,
                max_patience=50 * train_steps_per_epoch,
                minimize=False)
        ]

        serving_input_fn = generate_serving_input_fn(metadata)
        export_strategy = tf.contrib.learn.utils.make_export_strategy(
            serving_input_fn)

        experiment = tf.contrib.learn.Experiment(
            estimator=estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            eval_metrics=eval_metrics,
            train_monitors=train_monitors,
            train_steps=None,
            eval_steps=None,
            export_strategies=[export_strategy],
            min_eval_frequency=100)
        return experiment

    return _experiment_fn
