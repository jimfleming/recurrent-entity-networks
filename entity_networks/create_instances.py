from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import random
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from entity_networks.dataset import Dataset

def main():
    tasks_dir = 'tasks/'

    if not os.path.exists(tasks_dir):
        os.makedirs(tasks_dir)

    task_names = [
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

    for task_name in tqdm(task_name.iteritems()):
        dataset_path = 'data/records/{}_10k.json'.format(task_name)
        output_path = os.path.join(tasks_dir, '{}.json'.format(task_name))

        with tf.Graph().as_default():
            dataset = Dataset(dataset_path, 1)
            input_fn = dataset.get_input_fn('test', 1, shuffle=False)

            features, answer = input_fn()

            story = features['story']
            query = features['query']

            instances = []

            task = {}
            task['task_id'] = dataset.task_id
            task['task_name'] = dataset.task_name
            task['task_title'] = dataset.task_title
            task['max_query_length'] = dataset.max_query_length
            task['max_story_length'] = dataset.max_story_length
            task['max_sentence_length'] = dataset.max_sentence_length
            task['vocab'] = vocab.tolist()

            with tf.train.SingularMonitoredSession() as sess:
                while not sess.should_stop():
                    story_, query_, answer_ = sess.run([story, query, answer])

                    instance = {
                        'story': story_[0].tolist(),
                        'query': query_[0].tolist(),
                        'answer': answer_[0].tolist(),
                    }

                    instances.append(instance)

            task['instances'] = random.sample(instances, k=10)

            with open(output_path, 'w') as f:
                f.write(json.dumps(task))

if __name__ == '__main__':
    main()
