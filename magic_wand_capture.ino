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

#include "accelerometer_handler.h"
#include "constants.h"
#include "gesture_predictor.h"
#include "magic_wand_model_data.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Buffer to hold the input accelerometer samples before they're normalized
// for gravity.
float input_sample_buffer[kInputElementCount];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB
  }
  
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroMutableOpResolver<6> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor.
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != kInputBatchCount) ||
      (model_input->dims->data[1] != kInputSampleCount) ||
      (model_input->dims->data[2] != kInputChannelCount) ||
      (model_input->type != kTfLiteFloat32) ||
      (model_input->bytes != kInputByteCount)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Set up failed\n");
  }
}

bool IsMoving() {
  // Look at the most recent accelerometer values.
  const float* input_data = input_sample_buffer;
  const float last_x = input_data[kInputElementCount - 3];
  const float last_y = input_data[kInputElementCount - 2];
  const float last_z = input_data[kInputElementCount - 1];

  // Figure out the total amount of acceleration being felt by the device.
  const float last_x_squared = last_x * last_x;
  const float last_y_squared = last_y * last_y;
  const float last_z_squared = last_z * last_z;
  const float acceleration_magnitude =
      sqrtf(last_x_squared + last_y_squared + last_z_squared);

  // Acceleration is in milli-Gs, so normal gravity is 1,000 units.
  const float gravity = 1000.0f;

  // Subtract out gravity to get the actual movement magnitude.
  const float movement = acceleration_magnitude - gravity;

  // How much acceleration is needed before it's considered movement.
  const float movement_threshold = 40.0f;
  const bool is_moving = (movement > movement_threshold);

  return is_moving;
}

float VectorMagnitude(const float* vec) {
  const float x = vec[0];
  const float y = vec[1];
  const float z = vec[2];
  return sqrtf((x * x) + (y * y) + (z * z));
}

void NormalizeVector(const float* in_vec, float* out_vec) {
  const float magnitude = VectorMagnitude(in_vec);
  const float x = in_vec[0];
  const float y = in_vec[1];
  const float z = in_vec[2];
  out_vec[0] = x / magnitude;
  out_vec[1] = y / magnitude;
  out_vec[2] = z / magnitude;
}

float DotProduct(const float* a, const float* b) {
  return (a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

void EstimateGravityDirection(const float* sequence, int sequence_length, float* gravity) {
  float x_total = 0.0f;
  float y_total = 0.0f;
  float z_total = 0.0f;
  for (int i = 0; i < sequence_length; ++i) {
    const int sequence_index = (i * 3);
    const float* entry = &sequence[sequence_index];
    const float x = entry[0];
    const float y = entry[1];
    const float z = entry[2];
    x_total += x;
    y_total += y;
    z_total += z;
  }
  gravity[0] = x_total / sequence_length;
  gravity[1] = y_total / sequence_length;
  gravity[2] = z_total / sequence_length;
}

void RemoveGravityFromAccelerationData(const float* in_sequence, int sequence_length, float* out_sequence) {
  float gravity_direction[3];
  EstimateGravityDirection(in_sequence, sequence_length, gravity_direction);
  const float gravity_x = gravity_direction[0];
  const float gravity_y = gravity_direction[1];
  const float gravity_z = gravity_direction[2];
  for (int i = 0; i < sequence_length; ++i) {
    const int sequence_index = (i * 3);
    const float* in_entry = &in_sequence[sequence_index];
    const float x = in_entry[0];
    const float y = in_entry[1];
    const float z = in_entry[2];
    float* out_entry = &out_sequence[sequence_index];
    out_entry[0] = x - gravity_x;
    out_entry[1] = y - gravity_y;
    out_entry[2] = z - gravity_z;
  }
}

// This is the regular function we run to recognize gestures from a pretrained
// model.
void RecognizeGestures() {
  const bool is_moving = IsMoving();

  // Static state used to control the capturing process.
  static int counter = 0;
  static enum {
    ePendingStillness,
    eInStillness,
    ePendingMovement,
    eRecordingGesture
  } state = ePendingStillness;
  static int still_found_time;
  static int gesture_start_time;
  // State machine that controls gathering user input.
  switch (state) {
    case ePendingStillness: {
      if (!is_moving) {
        still_found_time = counter;
        state = eInStillness;
      }
    } break;

    case eInStillness: {
      if (is_moving) {
        state = ePendingStillness;
      } else {
        const int duration = counter - still_found_time;
        if (duration > 25) {
          state = ePendingMovement;
        }
      }
    } break;

    case ePendingMovement: {
      if (is_moving) {
        state = eRecordingGesture;
        gesture_start_time = counter;
      }
    } break;

    case eRecordingGesture: {
      const int recording_time = 128;
      if ((counter - gesture_start_time) > recording_time) {
        // Copy normalized data into the model's input buffer.
        float* model_input_buffer = interpreter->input(0)->data.f;
        RemoveGravityFromAccelerationData(input_sample_buffer, kInputSampleCount, model_input_buffer);
        
        // Run inference, and report any error.
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
          TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on index: %d\n",
                               begin_index);
          return;
        }

        const float* prediction_scores = interpreter->output(0)->data.f;
        TF_LITE_REPORT_ERROR(error_reporter, "Prediction: %d, %d, %d, %d",
                             (int)(prediction_scores[0] * 1000), 
                             (int)(prediction_scores[1] * 1000), 
                             (int)(prediction_scores[2] * 1000), 
                             (int)(prediction_scores[3] * 1000));
        const int found_gesture = PredictGesture(prediction_scores);

        // Produce an output
        HandleOutput(error_reporter, found_gesture);

        state = ePendingStillness;
      }
    } break;

    default: {
      TF_LITE_REPORT_ERROR(error_reporter, "Logic error - unknown state");
    } break;
  }
  
  // Increment the timing counter.
  ++counter;
}

// If you need to gather training data, call this function from the main loop
// and it will guide the user through contributing data.
// The output that's logged to the console can be fed into the Python training
// scripts for this example.
void CaptureGestureData() {
  const bool is_moving = IsMoving();

  // Static state used to control the capturing process.
  static int counter = 0;
  static int gesture_count = 0;
  static enum {
    eStarting,
    ePendingStillness,
    eInStillness,
    ePendingMovement,
    eRecordingGesture
  } state = eStarting;
  static int still_found_time;
  static int gesture_start_time;
  static const char* next_gesture = nullptr;
  // State machine that controls gathering user input.
  switch (state) {
    case eStarting: {
      if (!next_gesture || (strcmp(next_gesture, "other") == 0)) {
        next_gesture = "wing";
      } else if (strcmp(next_gesture, "wing") == 0) {
        next_gesture = "ring";
      } else if (strcmp(next_gesture, "ring") == 0) {
        next_gesture = "slope";
      } else {
        next_gesture = "other";
      }
      TF_LITE_REPORT_ERROR(error_reporter, "# Hold the wand still");
      state = ePendingStillness;
    } break;

    case ePendingStillness: {
      if (!is_moving) {
        still_found_time = counter;
        state = eInStillness;
      }
    } break;

    case eInStillness: {
      if (is_moving) {
        state = ePendingStillness;
      } else {
        const int duration = counter - still_found_time;
        if (duration > 25) {
          state = ePendingMovement;
          TF_LITE_REPORT_ERROR(error_reporter,
                               "# When you're ready, perform the %s gesture",
                               next_gesture);
        }
      }
    } break;

    case ePendingMovement: {
      if (is_moving) {
        state = eRecordingGesture;
        gesture_start_time = counter;
        TF_LITE_REPORT_ERROR(error_reporter, "# Perform the %s gesture now",
                             next_gesture);
      }
    } break;

    case eRecordingGesture: {
      const int recording_time = 100;
      if ((counter - gesture_start_time) > recording_time) {
        ++gesture_count;
        TF_LITE_REPORT_ERROR(error_reporter, "****************");
        TF_LITE_REPORT_ERROR(error_reporter, "gesture: %s", next_gesture);
        const float* input_data = input_sample_buffer;
        for (int offset = recording_time - 10; offset > 0; --offset) {
          const int array_offset = (kInputElementCount - (offset * 3));
          const int x = static_cast<int>(input_data[array_offset + 0]);
          const int y = static_cast<int>(input_data[array_offset + 1]);
          const int z = static_cast<int>(input_data[array_offset + 2]);
          TF_LITE_REPORT_ERROR(error_reporter, "x: %d y:%d z:%d", x, y, z);
        }
        TF_LITE_REPORT_ERROR(error_reporter, "~~~~~~~~~~~~~~~~");
        TF_LITE_REPORT_ERROR(error_reporter, "# %d gestures recorded",
                             gesture_count);
        state = eStarting;
      }
    } break;

    default: {
      TF_LITE_REPORT_ERROR(error_reporter, "Logic error - unknown state");
    } break;
  }

  // Increment the timing counter.
  ++counter;
}

void loop() {
  // Attempt to read new data from the accelerometer.
  bool got_data =
      ReadAccelerometer(error_reporter, input_sample_buffer, kInputElementCount);

  // If there was no new data, wait until next time.
  if (!got_data) return;

  // In the future we should decide whether to capture data based on a user
  // action (like pressing a button), but since some of the devices we're
  // targeting don't have any built-in input devices you'll need to manually
  // switch between recognizing gestures and capturing training data by changing
  // this variable and recompiling.
  const bool should_capture_data = false;
  if (should_capture_data) {
    CaptureGestureData();
  } else {
    RecognizeGestures();
  }
}
