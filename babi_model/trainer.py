from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tqdm import tqdm

class Trainer(object):

    def __init__(self, supervisor, sess, model_train, model_test):
        self.supervisor = supervisor
        self.sess = sess
        self.model_train = model_train
        self.model_test = model_test

    def train(self):
        history_size = 100

        loss_train_all = np.zeros(history_size)
        accuracy_train_all = np.zeros(history_size)

        loss_test_all = np.zeros(history_size)
        accuracy_test_all = np.zeros(history_size)

        try:
            with tqdm(desc='Training...') as pbar:
                step = 0
                while not self.supervisor.should_stop():
                    _, loss_train, accuracy_train = self.sess.run([
                        self.model_train.train_op,
                        self.model_train.loss,
                        self.model_train.accuracy
                    ])

                    loss_test, accuracy_test = self.sess.run([
                        self.model_test.loss,
                        self.model_test.accuracy
                    ])

                    loss_train_all[step%history_size] = loss_train
                    accuracy_train_all[step%history_size] = accuracy_train

                    loss_test_all[step%history_size] = loss_test
                    accuracy_test_all[step%history_size] = accuracy_test

                    loss_train_mean = np.mean(loss_train_all[:min(step, history_size)])
                    accuracy_train_mean = np.mean(accuracy_train_all[:min(step, history_size)])

                    loss_test_mean = np.mean(loss_test_all[:min(step, history_size)])
                    accuracy_test_mean = np.mean(accuracy_test_all[:min(step, history_size)])

                    pbar.set_description('[Train] Loss: {:.8f}, Accuracy: {:.2f} [Test] Loss: {:.8f}, Accuracy: {:.2f}' \
                        .format(loss_train_mean, accuracy_train_mean, loss_test_mean, accuracy_test_mean))
                    pbar.update()
                    step += 1
        except tf.errors.OutOfRangeError:
            print('Done.')
