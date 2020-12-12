# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-bad-import-order

"""Load data from the specified paths and format them for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np
import tensorflow as tf

from data_augmentation import augment_data

LABEL_NAME = "gesture"
HAND_LABEL_NAME = "hand_label"
DATA_NAME = "accel_ms2_xyz"


def vector_magnitude(vec):
  x = vec[0]
  y = vec[1]
  z = vec[2]
  return math.sqrt((x * x) + (y * y) + (z * z))


def normalize_vector(vec):
  magnitude = vector_magnitude(vec)
  x = vec[0]
  y = vec[1]
  z = vec[2]
  normalized_x = x / magnitude
  normalized_y = y / magnitude
  normalized_z = z / magnitude
  return (normalized_x, normalized_y, normalized_z)


def dot_product(a, b):
  return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def estimate_gravity_direction(sequence):
  x_total = 0.0
  y_total = 0.0
  z_total = 0.0
  for entry in sequence:
    x = entry[0]
    y = entry[1]
    z = entry[2]
    x_total += x
    y_total += y
    z_total += z
  return (x_total / len(sequence), y_total / len(sequence), z_total / len(sequence))


def remove_gravity_from_acceleration_data(sequence):
  gravity_direction = estimate_gravity_direction(sequence)
  gravity_x = gravity_direction[0] * 1.0
  gravity_y = gravity_direction[1] * 1.0
  gravity_z = gravity_direction[2] * 1.0
  result = []
  for entry in sequence:
    x = entry[0]
    y = entry[1]
    z = entry[2]
    normalized_x = x - gravity_x
    normalized_y = y - gravity_y
    normalized_z = z - gravity_z
    result.append([normalized_x, normalized_y, normalized_z])
  return result


class DataLoader(object):
  """Loads data and prepares for training."""

  def __init__(self, train_data_path, valid_data_path, test_data_path,
               seq_length):
    self.dim = 3
    self.seq_length = seq_length
    self.label2id = {"wing": 0, "ring": 1, "slope": 2, "negative": 3, "other": 3}
    self.train_data, self.train_label, self.train_len = self.get_data_file(
        train_data_path, "train")
    self.valid_data, self.valid_label, self.valid_len = self.get_data_file(
        valid_data_path, "valid")
    self.test_data, self.test_label, self.test_len = self.get_data_file(
        test_data_path, "test")

  def get_data_file(self, data_path, data_type):
    """Get train, valid and test data from files."""
    data = []
    label = []
    with open(data_path, "r") as f:
      lines = f.readlines()
      for idx, line in enumerate(lines):  # pylint: disable=unused-variable
        dic = json.loads(line)
        data.append(dic[DATA_NAME])
        if HAND_LABEL_NAME in dic:
          label.append(dic[HAND_LABEL_NAME])
        else:
          label.append(dic[LABEL_NAME])
    if data_type == "train":
      data, label = augment_data(data, label)
    length = len(label)
    print(data_type + "_data_length:" + str(length))
    return data, label, length

  def pad(self, data, seq_length, dim):
    """Get neighboor padding."""
    noise_level = 20
    padded_data = []
    # Before- Neighbour padding
    tmp_data = (np.random.rand(seq_length, dim) - 0.5) * noise_level + data[0]
    tmp_data[(seq_length -
              min(len(data), seq_length)):] = data[:min(len(data), seq_length)]
    padded_data.append(tmp_data)
    # After- Neighbour padding
    tmp_data = (np.random.rand(seq_length, dim) - 0.5) * noise_level + data[-1]
    tmp_data[:min(len(data), seq_length)] = data[:min(len(data), seq_length)]
    padded_data.append(tmp_data)
    return padded_data

  def format_support_func(self, padded_num, length, data, label):
    """Support function for format.(Helps format train, valid and test.)"""
    # Add 2 padding, initialize data and label
    length *= padded_num
    features = np.zeros((length, self.seq_length, self.dim))
    labels = np.zeros(length)
    # Get padding for train, valid and test
    for idx, (data, label) in enumerate(zip(data, label)):
      normalized_data = remove_gravity_from_acceleration_data(data)
      padded_data = self.pad(normalized_data, self.seq_length, self.dim)
      for num in range(padded_num):
        features[padded_num * idx + num] = padded_data[num]
        labels[padded_num * idx + num] = self.label2id[label.lower()]
    # Turn into tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (features, labels.astype("int32")))
    return length, dataset

  def format(self):
    """Format data(including padding, etc.) and get the dataset for the model."""
    padded_num = 2
    self.train_len, self.train_data = self.format_support_func(
        padded_num, self.train_len, self.train_data, self.train_label)
    self.valid_len, self.valid_data = self.format_support_func(
        padded_num, self.valid_len, self.valid_data, self.valid_label)
    self.test_len, self.test_data = self.format_support_func(
        padded_num, self.test_len, self.test_data, self.test_label)
