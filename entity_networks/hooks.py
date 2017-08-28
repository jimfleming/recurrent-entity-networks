from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorflow.python.training import basic_session_run_hooks

class EarlyStoppingHook(tf.train.SessionRunHook):

    def __init__(self, input_fn, estimator, metrics,
                 metric_name='loss', every_steps=100,
                 max_patience=100, minimize=True):
        self._input_fn = input_fn
        self._estimator = estimator
        self._metrics = metrics

        self._metric_name = metric_name
        self._every_steps = every_steps
        self._max_patience = max_patience
        self._minimize = minimize

        self._timer = basic_session_run_hooks.SecondOrStepTimer(
            every_steps=every_steps,
            every_secs=None)

        self._global_step = None
        self._best_value = None
        self._best_step = None

    def begin(self):
        self._global_step = tf.train.get_global_step()
        if self._global_step is None:
            raise RuntimeError('Global step should be created to use EarlyStoppingHook.')

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._global_step)

    def after_run(self, run_context, run_values):
        global_step = run_values.results

        if not self._timer.should_trigger_for_step(global_step):
            return

        self._timer.update_last_triggered_step(global_step)

        results = self._estimator.evaluate(
            input_fn=self._input_fn,
            metrics=self._metrics)

        if self._metric_name not in results:
            raise ValueError('Metric {} missing from outputs {}.' \
                .format(self._metric_name, set(results.keys())))

        current_value = results[self._metric_name]

        if (self._best_value is None) or \
           (self._minimize and current_value < self._best_value) or \
           (not self._minimize and current_value > self._best_value):
            self._best_value = current_value
            self._best_step = global_step

        should_stop = (global_step - self._best_step >= self._max_patience)
        if should_stop:
            print('Stopping... Best step: {} with {} = {}.' \
                .format(self._best_step, self._metric_name, self._best_value))
            run_context.request_stop()
