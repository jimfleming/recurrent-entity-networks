from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import re
import tarfile
import numpy as np
import tensorflow as tf

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

SPLIT_RE = re.compile('(\W+)?')

def tokenize(sentence):
    return [x.strip() for x in re.split(SPLIT_RE, sentence) if x.strip()]

def flatten_story(story):
    return [word for sentence in story for word in sentence]

def parse_stories(lines):
    """
    Parse stories provided in the bAbi tasks format
    """
    data = []
    story = []

    for line in lines:
        id_, line = line.split(' ', 1)
        id_ = int(id_)

        if id_ == 1:
            story = []

        if '\t' in line:
            q, a, _ = line.split('\t')
            q = tokenize(q)
            story_flat = flatten_story(story)
            data.append((story_flat, q, a))
        else:
            sentence = tokenize(line)
            story.append(sentence)

    return data

def print_dataset(data, count=3):
    for i, (story, q, a) in enumerate(data):
        if i > count - 1:
            break
        print('Story:', story)
        print('Q:', q)
        print('A:', a)

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

def main(_):
    # TODO: train baseline LSTM

    tar = tarfile.open('datasets/raw/babi_tasks_data_1_20_v1.2.tar.gz')

    file_train = tar.extractfile('tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt')
    file_test = tar.extractfile('tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt')

    dataset_train = parse_stories(file_train.readlines())
    dataset_test = parse_stories(file_test.readlines())

    EMBED_HIDDEN_SIZE = 50
    SENT_HIDDEN_SIZE = 100
    QUERY_HIDDEN_SIZE = 100
    BATCH_SIZE = 32
    EPOCHS = 40

    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in dataset_train + dataset_test)))

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in dataset_train + dataset_test)))
    query_maxlen = max(map(len, (x for _, x, _ in dataset_train + dataset_test)))

    X, Xq, Y = vectorize_stories(dataset_train, word_idx, story_maxlen, query_maxlen)
    tX, tXq, tY = vectorize_stories(dataset_test, word_idx, story_maxlen, query_maxlen)

    print('vocab = {}'.format(vocab))
    print('X.shape = {}'.format(X.shape))
    print('Xq.shape = {}'.format(Xq.shape))
    print('Y.shape = {}'.format(Y.shape))
    print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

    print('Build model...')
    sentrnn = Sequential()
    sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=story_maxlen))
    sentrnn.add(Dropout(0.3))

    qrnn = Sequential()
    qrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=query_maxlen))
    qrnn.add(Dropout(0.3))
    qrnn.add(LSTM(EMBED_HIDDEN_SIZE, return_sequences=False))
    qrnn.add(RepeatVector(story_maxlen))

    model = Sequential()
    model.add(Merge([sentrnn, qrnn], mode='sum'))
    model.add(LSTM(EMBED_HIDDEN_SIZE, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('Training...')
    model.fit([X, Xq], Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05)

    loss, acc = model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

if __name__ == '__main__':
    tf.app.run()
