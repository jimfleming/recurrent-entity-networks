"Training task script."
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from entity_networks.inputs import generate_input_fn
from entity_networks.serving import serving_input_fn
from entity_networks.model import model_fn

def generate_experiment_fn(data_dir, data_key, train_batch_size, eval_batch_size,
                           num_epochs, train_steps, eval_steps):
    "Return _experiment_fn for use with learn_runner."

    def _experiment_fn(model_dir):

        # data dir and key
        # look up filename from queue
        datasets = {
            'qa1': 'qa1_single-supporting-fact_10k.json',
        }

        # TODO: read metadata

        train_input_fn = generate_input_fn(
            filenames=train_files,
            batch_size=train_batch_size,
            num_epochs=num_epochs,
            shuffle=True)

        eval_input_fn = generate_input_fn(
            filenames=eval_files,
            batch_size=eval_batch_size,
            num_epochs=1,
            shuffle=False)

        run_config = tf.contrib.learn.RunConfig()

        # TODO: define hyperparameters, e.g. learning_rate
        params = {
            'vocab_size': dataset.vocab_size,
            'embedding_size': 100,
            'num_blocks': 20,
            'learning_rate_init': 1e-2,
            'learning_rate_decay_steps': dataset.steps_per_epoch * 25,
            'learning_rate_decay_rate': 0.5,
            'clip_gradients': 40.0,
        }

        estimator = tf.contrib.learn.Estimator(
            model_dir=model_dir,
            model_fn=model_fn,
            config=run_config,
            params=params,
            feature_engineering_fn=None)

        eval_metrics = {
            'accuracy': tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy)
        }

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
        '--train-files',
        help='Train files',
        nargs='+',
        required=True)
    parser.add_argument(
        '--eval-files',
        help='Evaluation files',
        nargs='+',
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

    tf.logging.set_verbosity(tf.logging.INFO)

    experiment_fn = generate_experiment_fn(
        train_files=args.train_files,
        eval_files=args.eval_files,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=args.num_epochs,
        train_steps=args.train_steps,
        eval_steps=args.eval_steps)
    tf.contrib.learn.learn_runner.run(experiment_fn, job_dir)

if __name__ == '__main__':
    main()
