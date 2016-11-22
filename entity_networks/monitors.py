from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tqdm import tqdm

class ProgressMonitor(tf.contrib.learn.monitors.EveryN):

    def __init__(self, tensor_names, every_n_steps=100, first_n_steps=1):
        super(ProgressMonitor, self).__init__(every_n_steps, first_n_steps)
        if not isinstance(tensor_names, dict):
            tensor_names = {tensor_name: tensor_name for tensor_name in tensor_names}
        self._tensor_names = tensor_names
        self._tensor_history = [np.zeros(every_n_steps) for tensor_name in tensor_names]
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
        for (tag, tensor_name), tensor_history in zip(self._tensor_names.iteritems(), self._tensor_history):
            tensor_history[step%self._every_n_steps] = outputs[tensor_name]
            tensor_mean = np.mean(tensor_history[:min(step, self._every_n_steps)])
            stats.append("{}: {:.6f}".format(tag, tensor_mean))
        self._progress_bar.set_description(", ".join(stats))
        self._progress_bar.update(step - self._last_step)
        self._last_step = step
