"Training task script."
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import argparse
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import learn_runner

from entity_networks.inputs import generate_input_fn
from entity_networks.serving import generate_serving_input_fn
from entity_networks.model import model_fn

SHARED_DIR = os.environ.get('SHARED_DIR', None)
RUN_NAME = os.environ.get('RUN_NAME', None)

def generate_experiment_fn(data_dir, dataset_id,
                           train_batch_size, eval_batch_size,
                           num_epochs, train_steps, eval_steps):
    "Return _experiment_fn for use with learn_runner."
    if SHARED_DIR is not None:
        data_dir = os.path.join(SHARED_DIR, data_dir)

    def _experiment_fn(output_dir):
        if SHARED_DIR is not None:
            output_dir = os.path.join(SHARED_DIR, 'runs', RUN_NAME, output_dir)

        metadata_path = os.path.join(data_dir, '{}_10k.json'.format(dataset_id))
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)

        train_filename = os.path.join(data_dir, '{}_10k_{}.tfrecords'.format(dataset_id, 'train'))
        eval_filename = os.path.join(data_dir, '{}_10k_{}.tfrecords'.format(dataset_id, 'test'))

        train_input_fn = generate_input_fn(
            filename=train_filename,
            metadata=metadata,
            batch_size=train_batch_size,
            num_epochs=num_epochs,
            shuffle=True)

        eval_input_fn = generate_input_fn(
            filename=eval_filename,
            metadata=metadata,
            batch_size=eval_batch_size,
            num_epochs=1,
            shuffle=False)

        run_config = tf.contrib.learn.RunConfig()

        vocab_size = metadata['vocab_size']
        task_size = metadata['task_size']
        train_steps_per_epoch = task_size // train_batch_size

        params = {
            'vocab_size': vocab_size,
            'embedding_size': 100,
            'num_blocks': 20,
            'learning_rate_init': 1e-2,
            'learning_rate_decay_steps': train_steps_per_epoch * 25,
            'learning_rate_decay_rate': 0.5,
            'clip_gradients': 40.0,
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

        serving_input_fn = generate_serving_input_fn(metadata)
        export_strategy = tf.contrib.learn.make_export_strategy(
            serving_input_fn=serving_input_fn)

        experiment = tf.contrib.learn.Experiment(
            estimator=estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            eval_metrics=eval_metrics,
            train_steps=train_steps,
            eval_steps=eval_steps,
            export_strategies=[export_strategy])
        return experiment

    return _experiment_fn

def main():
    "Entrypoint for training."
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-dir',
        help='Directory containing data',
        default='data/babi/records/',
        required=True)
    parser.add_argument(
        '--dataset-id',
        help='Unique id identifying dataset',
        required=True)
    parser.add_argument(
        '--job-dir',
        help='Location to write checkpoints, summaries, and export models',
        required=True)
    parser.add_argument(
        '--num-epochs',
        help='Maximum number of epochs on which to train',
        default=200,
        type=int)
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=32)
    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=32)
    parser.add_argument(
        '--train-steps',
        help='Number of steps to run training',
        type=int)
    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evaluation at each checkpoint',
        type=int)

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    experiment_fn = generate_experiment_fn(
        data_dir=args.data_dir,
        dataset_id=args.dataset_id,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        train_steps=args.train_steps,
        eval_steps=args.eval_steps)
    learn_runner.run(experiment_fn, args.job_dir)

if __name__ == '__main__':
    main()
