# Copyright 2020 Jack Spencer Smith.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_filename', None, 'Path to input tfrecord')
flags.DEFINE_string('output_directory', None, 'Path to output_directory')
flags.DEFINE_integer('num_shards', 12, 'number of examples per shard')
flags.DEFINE_integer('total_size', 305979, 'total size of dataset')
flags.DEFINE_string('expected_splits', 'train,validation,test',
                    'Comma separated list of expected splits.')
flags.DEFINE_list(
    'pipeline_options', '--runner=DirectRunner',
    'A comma-separated list of command line arguments to be used as options '
    'for the Beam Pipeline.')


def reduce_func(key, dataset):
    filename = tf.strings.join([FLAGS.output_directory, tf.strings.as_string(key)])
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)


def shard(argv):
    del argv
    flags.mark_flags_as_required(['input_filename', 'output_directory'])

    dataset = tf.data.TFRecordDataset(FLAGS.input_filename)
    for i in range(FLAGS.num_shards):
        sharded = dataset.shard(FLAGS.num_shards, i)
        # completely shuffle them
        sharded = sharded.shuffle(int(FLAGS.total_size / (FLAGS.num_shards - 1)))
        filename = tf.strings.join([FLAGS.output_directory, tf.strings.as_string(i)])
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(sharded)


if __name__ == '__main__':
    app.run(shard)
