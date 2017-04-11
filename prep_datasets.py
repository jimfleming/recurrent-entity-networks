"""
This module loads and pre-processes a bAbI dataset into TFRecords.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import re
import json
import tarfile
import numpy as np
import tensorflow as tf

from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('source_dir', 'datasets/', 'Directory containing bAbI sources.')
tf.app.flags.DEFINE_string('dest_dir', 'datasets/processed/', 'Where to write datasets.')
tf.app.flags.DEFINE_boolean('include_10k', True, 'Whether to use 10k or 1k examples.')

SPLIT_RE = re.compile('(\W+)?')

PAD_TOKEN = '_PAD'
PAD_ID = 0

def int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def tokenize(sentence):
    """
    Tokenize a string by splitting on non-word characters and stripping whitespace.
    """
    return [token.strip().lower() for token in re.split(SPLIT_RE, sentence) if token.strip()]

def parse_stories(lines, only_supporting=False):
    """
    Parse the bAbI task format described here: https://research.facebook.com/research/babi/
    If only_supporting is True, only the sentences that support the answer are kept.
    """
    stories = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            query, answer, supporting = line.split('\t')
            query = tokenize(query)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            stories.append((substory, query, answer))
            story.append('')
        else:
            sentence = tokenize(line)
            story.append(sentence)
    return stories

def save_dataset(stories, path):
    """
    Save the stories into TFRecords.

    NOTE: Since each sentence is a consistent length from padding, we use
    `tf.train.Example`, rather than a `tf.train.SequenceExample`, which is
    _slightly_ faster.
    """
    writer = tf.python_io.TFRecordWriter(path)
    for story, query, answer in stories:
        story_flat = [token_id for sentence in story for token_id in sentence]

        features = tf.train.Features(feature={
            'story': int64_features(story_flat),
            'query': int64_features(query),
            'answer': int64_features([answer]),
        })

        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()

def tokenize_stories(stories, token_to_id):
    """
    Convert all tokens into their unique ids.
    """
    story_ids = []
    for story, query, answer in stories:
        story = [[token_to_id[token] for token in sentence] for sentence in story]
        query = [token_to_id[token] for token in query]
        answer = token_to_id[answer]
        story_ids.append((story, query, answer))
    return story_ids

def get_tokenizer(stories):
    """
    Recover unique tokens as a vocab and map the tokens to ids.
    """
    tokens_all = []
    for story, query, answer in stories:
        tokens_all.extend([token for sentence in story for token in sentence] + query + [answer])
    vocab = [PAD_TOKEN] + sorted(set(tokens_all))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    return token_to_id

def pad_stories(stories, max_sentence_length, max_story_length, max_query_length):
    """
    Pad sentences, stories, and queries to a consistence length.
    """
    for story, query, answer in stories:
        for sentence in story:
            for _ in range(max_sentence_length - len(sentence)):
                sentence.append(PAD_ID)
            assert len(sentence) == max_sentence_length

        for _ in range(max_story_length - len(story)):
            story.append([PAD_ID for _ in range(max_sentence_length)])

        for _ in range(max_query_length - len(query)):
            query.append(PAD_ID)

        assert len(story) == max_story_length
        assert len(query) == max_query_length

    return stories

def truncate_stories(stories, max_length):
    stories_truncated = []
    for story, query, answer in stories:
        story_truncated = story[-max_length:]
        stories_truncated.append((story_truncated, query, answer))
    return stories_truncated

def main():
    if not os.path.exists(FLAGS.dest_dir):
        os.makedirs(FLAGS.dest_dir)

    filenames = [
        'qa1_single-supporting-fact',
        'qa2_two-supporting-facts',
        'qa3_three-supporting-facts',
        'qa4_two-arg-relations',
        'qa5_three-arg-relations',
        'qa6_yes-no-questions',
        'qa7_counting',
        'qa8_lists-sets',
        'qa9_simple-negation',
        'qa10_indefinite-knowledge',
        'qa11_basic-coreference',
        'qa12_conjunction',
        'qa13_compound-coreference',
        'qa14_time-reasoning',
        'qa15_basic-deduction',
        'qa16_basic-induction',
        'qa17_positional-reasoning',
        'qa18_size-reasoning',
        'qa19_path-finding',
        'qa20_agents-motivations',
    ]

    tar = tarfile.open(os.path.join(FLAGS.source_dir, 'babi_tasks_data_1_20_v1.2.tar.gz'))
    for filename in tqdm(filenames):
        if FLAGS.include_10k:
            stories_path_train = os.path.join('tasks_1-20_v1-2/en-10k/', filename + '_train.txt')
            stories_path_test = os.path.join('tasks_1-20_v1-2/en-10k/', filename + '_test.txt')
            dataset_path_train = os.path.join(FLAGS.dest_dir, filename + '_10k_train.tfrecords')
            dataset_path_test = os.path.join(FLAGS.dest_dir, filename + '_10k_test.tfrecords')
            metadata_path = os.path.join(FLAGS.dest_dir, filename + '_10k.json')
            dataset_size = 10000
        else:
            stories_path_train = os.path.join('tasks_1-20_v1-2/en/', filename + '_train.txt')
            stories_path_test = os.path.join('tasks_1-20_v1-2/en/', filename + '_test.txt')
            dataset_path_train = os.path.join(FLAGS.dest_dir, filename + '_1k_train.tfrecords')
            dataset_path_test = os.path.join(FLAGS.dest_dir, filename + '_1k_test.tfrecords')
            metadata_path = os.path.join(FLAGS.dest_dir, filename + '_1k.json')
            dataset_size = 1000

        # From the entity networks paper:
        # > Copying previous works (Sukhbaatar et al., 2015; Xiong et al., 2016), the capacity of the memory
        # > was limited to the most recent 70 sentences, except for task 3 which was limited to 130 sentences.
        if filename == 'qa3_three-supporting-facts':
            truncated_story_length = 130
        else:
            truncated_story_length = 70

        f_train = tar.extractfile(stories_path_train)
        f_test = tar.extractfile(stories_path_test)

        stories_train = parse_stories(f_train.readlines())
        stories_test = parse_stories(f_test.readlines())

        stories_train = truncate_stories(stories_train, truncated_story_length)
        stories_test = truncate_stories(stories_test, truncated_story_length)

        token_to_id = get_tokenizer(stories_train + stories_test)

        stories_token_train = tokenize_stories(stories_train, token_to_id)
        stories_token_test = tokenize_stories(stories_test, token_to_id)
        stories_token_all = stories_token_train + stories_token_test

        max_sentence_length = max([len(sentence) for story, _, _ in stories_token_all for sentence in story])
        max_story_length = max([len(story) for story, _, _ in stories_token_all])
        max_query_length = max([len(query) for _, query, _ in stories_token_all])
        vocab_size = len(token_to_id)

        with open(metadata_path, 'w') as f:
            metadata = {
                'dataset_name': filename,
                'dataset_size': dataset_size,
                'max_sentence_length': max_sentence_length,
                'max_story_length': max_story_length,
                'max_query_length': max_query_length,
                'vocab_size': vocab_size,
                'tokens': token_to_id,
                'datasets': {
                    'train': os.path.basename(dataset_path_train),
                    'test': os.path.basename(dataset_path_test),
                }
            }
            json.dump(metadata, f)

        stories_pad_train = pad_stories(stories_token_train, \
            max_sentence_length, max_story_length, max_query_length)
        stories_pad_test = pad_stories(stories_token_test, \
            max_sentence_length, max_story_length, max_query_length)

        save_dataset(stories_pad_train, dataset_path_train)
        save_dataset(stories_pad_test, dataset_path_test)

if __name__ == '__main__':
    main()
