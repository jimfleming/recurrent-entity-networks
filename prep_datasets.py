from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import re
import tarfile
import numpy as np
import tensorflow as tf

from tqdm import tqdm

SPLIT_RE = re.compile('(\W+)?')

def tokenize(sentence):
    return [x.strip() for x in re.split(SPLIT_RE, sentence) if x.strip()]

def parse_stories(lines):
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
            data.append((story, q, a))
        else:
            sentence = tokenize(line)
            story.append(sentence)
    return data

def save_dataset(stories, sentence_max_length, story_max_length, query_max_length, path):
    writer = tf.python_io.TFRecordWriter(path)
    for story, query, answer in tqdm(stories):
        story_length = len(story)
        query_length = len(query)

        example = tf.train.SequenceExample()
        example.context.feature['answer'].int64_list.value.append(answer)

        story_list = example.feature_lists.feature_list['story']
        query_list = example.feature_lists.feature_list['query']

        for sentence in story:
            for token_id in sentence:
                story_list.feature.add().int64_list.value.append(token_id)
            for _ in range(sentence_max_length - len(sentence)):
                story_list.feature.add().int64_list.value.append(0)

        for token_id in query:
            query_list.feature.add().int64_list.value.append(token_id)

        # pre-wrap and pad sentences to consistent length, then flatten
        for _ in range(story_max_length - story_length):
            for _ in range(sentence_max_length):
                story_list.feature.add().int64_list.value.append(0)

        for _ in range(query_max_length - query_length):
            query_list.feature.add().int64_list.value.append(0)

        writer.write(example.SerializeToString())
    writer.close()

def tokenize_stories(stories, token_to_id):
    story_ids = []
    for story, query, answer in stories:
        story = [[token_to_id[token] for token in sentence] for sentence in story]
        query = [token_to_id[token] for token in query]
        answer = token_to_id[answer]
        story_ids.append((story, query, answer))
    return story_ids

def main():
    filenames = [
        'qa1_single-supporting-fact',
        # 'qa2_two-supporting-facts',
        # 'qa3_three-supporting-facts',
        # 'qa4_two-arg-relations',
        # 'qa5_three-arg-relations',
        # 'qa6_yes-no-questions',
        # 'qa7_counting',
        # 'qa8_lists-sets',
        # 'qa9_simple-negation',
        # 'qa10_indefinite-knowledge',
        # 'qa11_basic-coreference',
        # 'qa12_conjunction',
        # 'qa13_compound-coreference',
        # 'qa14_time-reasoning',
        # 'qa15_basic-deduction',
        # 'qa16_basic-induction',
        # 'qa17_positional-reasoning',
        # 'qa18_size-reasoning',
        # 'qa19_path-finding',
        # 'qa20_agents-motivations',
    ]

    tar = tarfile.open('datasets/babi_tasks_data_1_20_v1.2.tar.gz')
    for filename in tqdm(filenames):
        stories_path_train = os.path.join('tasks_1-20_v1-2/en-10k/', filename + '_train.txt')
        stories_path_test = os.path.join('tasks_1-20_v1-2/en-10k/', filename + '_test.txt')

        dataset_path_train = os.path.join('datasets/processed/', filename + '_train.tfrecords')
        dataset_path_test = os.path.join('datasets/processed/', filename + '_test.tfrecords')

        f_train = tar.extractfile(stories_path_train)
        f_test = tar.extractfile(stories_path_test)

        stories_train = parse_stories(f_train.readlines())
        stories_test = parse_stories(f_test.readlines())
        stories_all = stories_train + stories_test

        tokens_all = []
        for story, query, answer in stories_all:
            tokens_all.extend([token for sentence in story for token in sentence] + query + [answer])
        vocab = sorted(set(tokens_all))
        vocab_size = len(vocab) + 1 # reserve zero for padding
        token_to_id = {token: i+1 for i, token in enumerate(vocab)}

        stories_train = tokenize_stories(stories_train, token_to_id)
        stories_test = tokenize_stories(stories_test, token_to_id)
        stories_all = stories_train + stories_test

        sentence_max_length = max([len(sentence) for story, _, _ in stories_all for sentence in story])

        story_max_length = max([len(story) for story, _, _ in stories_all])
        query_max_length = max([len(query) for _, query, _ in stories_all])

        print('Dataset:', filename)
        print('Max sentence length:', sentence_max_length)
        print('Max story length:', story_max_length)
        print('Max query length:', query_max_length)
        print('Vocab size:', vocab_size)

        save_dataset(stories_train, sentence_max_length, story_max_length, query_max_length, dataset_path_train)
        save_dataset(stories_test, sentence_max_length, story_max_length, query_max_length, dataset_path_test)

if __name__ == '__main__':
    main()
