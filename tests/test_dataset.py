from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json
import tensorflow as tf

from entity_networks.dataset import Dataset

class DatasetTest(tf.test.TestCase):

    def test_dataset(self):
        with self.test_session() as sess:
            dataset = Dataset(
                filename='datasets/processed/qa1_single-supporting-fact_train.tfrecords',
                batch_size=1,
                shuffle=False)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord, daemon=False)

            story_batch, query_batch, answer_batch = sess.run([
                dataset.story_batch,
                dataset.query_batch,
                dataset.answer_batch,
            ])

            coord.request_stop()
            coord.join(threads)

            with open('tokens.json') as f:
                token_to_id = json.load(f)
                id_to_token = {id_: token for token, id_ in token_to_id.iteritems()}

            story = story_batch[0]
            query = query_batch[0]
            answer = answer_batch[0]

            story_str = ' '.join([id_to_token[id_] for sentence in story for id_ in sentence if id_ != 0])
            query_str = ' '.join([id_to_token[id_] for sentence in query for id_ in sentence if id_ != 0])
            answer_str = id_to_token[answer]

            assert story_str == 'Mary moved to the bathroom . John went to the hallway .'
            assert query_str == 'Where is Mary ?'
            assert answer_str == 'bathroom'

if __name__ == '__main__':
    tf.test.main()
