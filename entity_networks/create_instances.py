from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from entity_networks.dataset import Dataset

def main():
    tasks_dir = 'tasks/'

    if not os.path.exists(tasks_dir):
        os.makedirs(tasks_dir)

    filenames = {
        'Task 1: Single Supporting Fact': 'qa1_single-supporting-fact',
        'Task 2: Two Supporting Facts': 'qa2_two-supporting-facts',
        'Task 3: Three Supporting Facts': 'qa3_three-supporting-facts',
        'Task 4: Two Argument Relations': 'qa4_two-arg-relations',
        'Task 5: Three Argument Relations': 'qa5_three-arg-relations',
        'Task 6: Yes/No Questions': 'qa6_yes-no-questions',
        'Task 7: Counting': 'qa7_counting',
        'Task 8: Lists/Sets': 'qa8_lists-sets',
        'Task 9: Simple Negation': 'qa9_simple-negation',
        'Task 10: IndefiniteKnowledg': 'qa10_indefinite-knowledge',
        'Task 11: Basic Coreference': 'qa11_basic-coreference',
        'Task 12: Conjunction': 'qa12_conjunction',
        'Task 13: Compound Coreference': 'qa13_compound-coreference',
        'Task 14: Time Reasoning': 'qa14_time-reasoning',
        'Task 15: Basic Deduction': 'qa15_basic-deduction',
        'Task 16: Basic Induction': 'qa16_basic-induction',
        'Task 17: Positional Reasoning': 'qa17_positional-reasoning',
        'Task 18: Size Reasoning': 'qa18_size-reasoning',
        'Task 19: Path Finding': 'qa19_path-finding',
        'Task 20: Agent Motivations': 'qa20_agents-motivations',
    }

    for task_name, filename in tqdm(filenames.iteritems()):
        with tf.Graph().as_default():
            dataset_path = 'data/records/{}_10k.json'.format(filename)
            output_path = os.path.join(tasks_dir, '{}.json'.format(filename))

            dataset = Dataset(dataset_path, 1)
            input_fn = dataset.get_input_fn('test', 1, shuffle=False)

            features, answer = input_fn()

            story = features['story']
            query = features['query']

            vocab = np.zeros(
                shape=dataset.vocab_size,
                dtype=object)

            for token, token_id in dataset.tokens.iteritems():
                vocab[token_id] = token

            task = {}
            task['name'] = task_name
            task['max_query_length'] = dataset.max_query_length
            task['max_story_length'] = dataset.max_story_length
            task['max_sentence_length'] = dataset.max_sentence_length
            task['instances'] = []
            task['vocab'] = vocab.tolist()

            with tf.train.SingularMonitoredSession() as sess:
                while len(task['instances']) < 10:
                    story_, query_, answer_ = sess.run([story, query, answer])

                    instance = {
                        'story': story_[0].tolist(),
                        'query': query_[0].tolist(),
                        'answer': answer_[0].tolist(),
                    }

                    task['instances'].append(instance)

            with open(output_path, 'w') as f:
                f.write(json.dumps(task))

if __name__ == '__main__':
    main()
