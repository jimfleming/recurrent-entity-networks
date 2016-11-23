from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from entity_networks.dataset import Dataset

class DatasetTest(tf.test.TestCase):

    def test_dataset(self):
        with self.test_session() as sess:
            dataset = Dataset(
                filenames=['datasets/processed/qa1_single-supporting-fact_train.tfrecords'],
                batch_size=3,
                shuffle=False)

            tf.train.start_queue_runners(sess)

            story_batch = dataset.story_batch.eval()
            query_batch = dataset.query_batch.eval()
            answer_batch = dataset.answer_batch.eval()

            with open('tokens.json') as f:
                token_to_id = json.load(f)
                id_to_token = {id_: token for token, id_ in token_to_id.iteritems()}

            print([[id_to_token[id_] for id_ in story] for story in story_batch])
            print([[id_to_token[id_] for id_ in query] for query in query_batch])
            print([[id_to_token[id_] for id_ in answer]])

if __name__ == '__main__':
    tf.test.main()
