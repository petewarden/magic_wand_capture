/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MAGIC_WAND_CONSTANTS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MAGIC_WAND_CONSTANTS_H_

// The expected accelerometer data sample frequency
const float kTargetHz = 25;

// Shape of the model input buffer.
constexpr int kInputBatchCount = 1;
constexpr int kInputSampleCount = 128;
constexpr int kInputChannelCount = 3;
constexpr int kInputElementCount = (kInputBatchCount * kInputSampleCount * kInputChannelCount);
constexpr int kInputByteCount = kInputElementCount * sizeof(float);

// What gestures are supported.
constexpr int kGestureCount = 4;
constexpr int kWingGesture = 0;
constexpr int kRingGesture = 1;
constexpr int kSlopeGesture = 2;
constexpr int kNoGesture = 3;

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MAGIC_WAND_CONSTANTS_H_
