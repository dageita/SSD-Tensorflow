# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Convert a dataset to TFRecords format, which can be easily integrated into
a TensorFlow pipeline.
NOTE that this script will generate a serises of tfrecords under dataset_dir folder \
    according to the total dataset number and parameter samples_per_files, \
    the formula is: the number of generated tfrecords = (total number // samples_per_files) + 1. \
    Please do not change dataset_name. \
    dataset_dir contains Annotations and JPEGImages.


Usage:
python tf_convert_data.py \
--dataset_name=pascalvoc \
--dataset_dir=/path/to/folder/ \
--output_name=voc_2007_test \
--output_dir=/path/to/folder/ \
--samples_per_files=200 \
--shuffling=True \
--class_names='background',...

NOTE 
--dataset_dir and --output_dir are path to folders, both string should end with '/'
--dataset_name: should be pascalvoc, do not change it.
--output_name: if train voc fromat dataset, this argument should be one of the following arguments \
    {voc_2007_, voc_2012_} + {train, test}. \
    As cifar10 and imagenet format dataset, please see relevant code.
"""
import tensorflow as tf

from datasets import pascalvoc_to_tfrecords

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc',
    'The name of the dataset to convert.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'Directory where the original dataset is stored.')
tf.app.flags.DEFINE_string(
    'output_name', 'pascalvoc',
    'Basename used for TFRecords output files.')
tf.app.flags.DEFINE_string(
    'output_dir', './',
    'Output directory where to store TFRecords files.')
tf.app.flags.DEFINE_integer(
    'samples_per_files', 200,
    'samples appointed number of dataset images to form a tfrecord, finally number of tfrecords = (total images // samples_per_files)+1.')
tf.app.flags.DEFINE_boolean(
    'shuffling', False,
    'Whether shuffle dataset.')
tf.app.flags.DEFINE_list(
    'class_names', ['background'],
    'the class names of your dataset, includes backgroud.')

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    print('Dataset directory:', FLAGS.dataset_dir)
    print('Output directory:', FLAGS.output_dir)

    if FLAGS.dataset_name == 'pascalvoc':
        pascalvoc_to_tfrecords.run(FLAGS.dataset_dir, FLAGS.output_dir, FLAGS.output_name, FLAGS.samples_per_files, FLAGS.shuffling,FLAGS.class_names)
    else:
        raise ValueError('Dataset [%s] was not recognized.' % FLAGS.dataset_name)

if __name__ == '__main__':
    tf.app.run()

