from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tqdm import tqdm

class ProgressMonitor(tf.contrib.learn.monitors.EveryN):

    def __init__(self, tensor_names, every_n_steps=1, first_n_steps=1, decay_rate=0.9):
        super(ProgressMonitor, self).__init__(every_n_steps, first_n_steps)
        if not isinstance(tensor_names, dict):
            tensor_names = {tensor_name: tensor_name for tensor_name in tensor_names}
        self._tensor_names = tensor_names
        self._decay_rate = decay_rate
        self._history = {tensor_name: 0.0 for tensor_name in tensor_names}
        self._last_step = 0

    def begin(self, max_steps=None):
        super(ProgressMonitor, self).begin(max_steps)
        self._progress_bar = tqdm(total=max_steps, unit='batches')

    def end(self, session=None):
        super(ProgressMonitor, self).end(session)
        self._progress_bar.close()

    def every_n_step_begin(self, step):
        super(ProgressMonitor, self).every_n_step_begin(step)
        return list(self._tensor_names.values())

    def every_n_step_end(self, step, outputs):
        super(ProgressMonitor, self).every_n_step_end(step, outputs)

        stats = []
        for tag, name in self._tensor_names.iteritems():
            if self._last_step > 0:
                self._history[name] = outputs[name] * (1 - self._decay_rate) + \
                    self._history[name] * self._decay_rate
            else:
                self._history[name] = outputs[name]
            stats.append("{}: {:.6f}".format(tag, self._history[name]))

        self._progress_bar.set_description(", ".join(stats))
        self._progress_bar.update(step - self._last_step)
        self._last_step = step
