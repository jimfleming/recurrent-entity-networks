"Training task script."
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import learn_runner

from entity_networks.experiment import generate_experiment_fn

def main():
    "Entrypoint for training."
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-dir',
        help='Directory containing data',
        default='data/babi/records/')
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
        '--lr-min',
        help='Minimum learning rate',
        default=2e-4,
        type=float)
    parser.add_argument(
        '--lr-max',
        help='Maximum learning rate',
        default=1e-2,
        type=float)
    parser.add_argument(
        '--lr-step-size',
        help='Learning rate step size (in epochs)',
        default=10,
        type=int)
    parser.add_argument(
        '--grad-noise',
        help='Gradient noise scale',
        default=0.005,
        type=float)

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    experiment_fn = generate_experiment_fn(
        data_dir=args.data_dir,
        dataset_id=args.dataset_id,
        num_epochs=args.num_epochs,
        learning_rate_min=args.lr_min,
        learning_rate_max=args.lr_max,
        learning_rate_step_size=args.lr_step_size,
        gradient_noise_scale=args.grad_noise)
    learn_runner.run(experiment_fn, args.job_dir)

if __name__ == '__main__':
    main()
