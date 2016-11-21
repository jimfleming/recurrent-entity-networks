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
        learning_rate_max = 1e-2

        num_epochs = 200
        batch_size = 32
        epoch_decay_rate = 25
        num_samples_per_epoch = 10000

        num_batches_per_epoch = num_samples_per_epoch // batch_size
        num_batches_per_decay = num_batches_per_epoch * epoch_decay_rate

        history_size = num_batches_per_epoch

        loss_train_all = np.zeros(history_size)
        accuracy_train_all = np.zeros(history_size)

        loss_test_all = np.zeros(history_size)
        accuracy_test_all = np.zeros(history_size)

        try:
            total_steps = num_epochs * num_batches_per_epoch
            with tqdm(desc='Training...', total=total_steps) as pbar:
                step = 0
                while not self.supervisor.should_stop():
                    learning_rate = learning_rate_max / 2**(step // num_batches_per_decay)
                    _, loss_train, accuracy_train, loss_test, accuracy_test = self.sess.run([
                        self.model_train.train_op,
                        self.model_train.loss,
                        self.model_train.accuracy,
                        self.model_test.loss,
                        self.model_test.accuracy
                    ], feed_dict={
                        self.model_train.learning_rate: learning_rate
                    })

                    loss_train_all[step%history_size] = loss_train
                    accuracy_train_all[step%history_size] = accuracy_train

                    loss_test_all[step%history_size] = loss_test
                    accuracy_test_all[step%history_size] = accuracy_test

                    if step > 1:
                        loss_train_mean = np.mean(loss_train_all[:min(step, history_size)])
                        accuracy_train_mean = np.mean(accuracy_train_all[:min(step, history_size)])

                        loss_test_mean = np.mean(loss_test_all[:min(step, history_size)])
                        accuracy_test_mean = np.mean(accuracy_test_all[:min(step, history_size)])

                        pbar.set_description('[Train] Loss: {:.8f}, Accuracy: {:.4f} [Test] Loss: {:.8f}, Accuracy: {:.4f} (Learning Rate: {:.8f})' \
                            .format(loss_train_mean, accuracy_train_mean, loss_test_mean, accuracy_test_mean, learning_rate))

                    pbar.update()
                    step += 1
        except tf.errors.OutOfRangeError:
            print('Done.')
