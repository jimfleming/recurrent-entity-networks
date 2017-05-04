from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import random
import argparse
import tensorflow as tf

from tqdm import tqdm

from entity_networks.inputs import generate_input_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        help='Directory containing data',
        default='data/babi/records/')
    args = parser.parse_args()

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
        metadata_path = os.path.join(args.data_dir, '{}_10k.json'.format(task_name))
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)

        filename = os.path.join(data_dir, '{}_10k_{}.tfrecords'.format(dataset_id, 'test'))
        input_fn = generate_input_fn(
            filename=eval_filename,
            metadata=metadata,
            batch_size=BATCH_SIZE,
            num_epochs=1,
            shuffle=False)

        with tf.Graph().as_default():
            features, answer = input_fn()

            story = features['story']
            query = features['query']

            instances = []

            with tf.train.SingularMonitoredSession() as sess:
                while not sess.should_stop():
                    story_, query_, answer_ = sess.run([story, query, answer])

                    instance = {
                        'story': story_[0].tolist(),
                        'query': query_[0].tolist(),
                        'answer': answer_[0].tolist(),
                    }

                    instances.append(instance)

            metadata['instances'] = random.sample(instances, k=10)

            output_path = os.path.join(tasks_dir, '{}.json'.format(task_name))
            with open(output_path, 'w') as f:
                f.write(json.dumps(metadata))

if __name__ == '__main__':
    main()
