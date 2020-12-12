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

"""Data augmentation that will be used in data_load.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np


def time_wrapping(molecule, denominator, data):
  """Generate (molecule/denominator)x speed data."""
  tmp_data = [[0
               for i in range(len(data[0]))]
              for j in range((int(len(data) / molecule) - 1) * denominator)]
  for i in range(int(len(data) / molecule) - 1):
    for j in range(len(data[i])):
      for k in range(denominator):
        tmp_data[denominator * i +
                 k][j] = (data[molecule * i + k][j] * (denominator - k) +
                          data[molecule * i + k + 1][j] * k) / denominator
  return tmp_data


def conditionally_add(new_data, new_label, data, label, add_probability):
  if random.random() < add_probability:
    new_data.append(data)
    new_label.append(label)


def augment_data(original_data, original_label):
  """Perform data augmentation."""
  label_counts = {}
  for (data, label) in zip(original_data, original_label):
    if label not in label_counts:
      label_counts[label] = 0
    label_counts[label] += 1
  total_label_count = len(original_data)
  new_data = []
  new_label = []
  for idx, (data, label) in enumerate(zip(original_data, original_label)):  # pylint: disable=unused-variable
    add_probability = total_label_count / label_counts[label]
    # Original data
    conditionally_add(new_data, new_label, data, label, add_probability)
    # Sequence shift
    for num in range(5):  # pylint: disable=unused-variable
      data = (np.array(data, dtype=np.float32) +
              (random.random() - 0.5) * 200).tolist()
      conditionally_add(new_data, new_label, data, label, add_probability)
    # Random noise
    tmp_data = [[0 for i in range(len(data[0]))] for j in range(len(data))]
    for num in range(5):
      for i in range(len(tmp_data)):
        for j in range(len(tmp_data[i])):
          tmp_data[i][j] = data[i][j] + 5 * random.random()
      conditionally_add(new_data, new_label, tmp_data, label, add_probability)
      new_data.append(tmp_data)
      new_label.append(label)
    # Time warping
    fractions = [(3, 2), (5, 3), (2, 3), (3, 4), (9, 5), (6, 5), (4, 5)]
    for molecule, denominator in fractions:
      data = time_wrapping(molecule, denominator, data)
      conditionally_add(new_data, new_label, data, label, add_probability)
    # Movement amplification
    for molecule, denominator in fractions:
      data = (np.array(data, dtype=np.float32) * molecule / denominator).tolist()
      conditionally_add(new_data, new_label, data, label, add_probability)
  return new_data, new_label
