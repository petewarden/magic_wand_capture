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

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "accelerometer_handler.h"
#include "constants.h"
#include "gesture_predictor.h"
#include "magic_wand_model_data.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {

// For accelerometer data.
constexpr int kSensorArenaSize = 3 * 200;
float sensor_arena[kSensorArenaSize];

tflite::ErrorReporter* error_reporter = nullptr;
bool should_clear_buffer = true;

int counter = 0;
enum { eWaitingForUpright, ePendingUpright, eIsUpright, eRecordingGesture, eStarting} state = eStarting;
int upright_found_time;
int gesture_start_time;
char* next_gesture = nullptr;

}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
  }
}

void loop() {
  const int input_length = 384;
  // Attempt to read new data from the accelerometer.
  bool got_data = ReadAccelerometer(error_reporter, sensor_arena,
                                    input_length, should_clear_buffer);
  
  // Don't try to clear the buffer again
  should_clear_buffer = false;
  // If there was no new data, wait until next time.
  if (!got_data) return;

  const int last_x = int(sensor_arena[input_length - 3]); 
  const int last_y = int(sensor_arena[input_length - 2]); 
  const int last_z = int(sensor_arena[input_length - 1]); 

  const bool is_upright = ((fabsf(last_x) < 100) && (fabsf(last_y) < 100) && (last_z > 950));
  
  switch (state) {
    case eStarting: {
      if (!next_gesture || (strcmp(next_gesture, "slope") == 0)) {
        next_gesture = "wing";
      } else if (strcmp(next_gesture, "wing") == 0) {
        next_gesture = "ring";
      } else if (strcmp(next_gesture, "ring") == 0) {
        next_gesture = "slope";
      } else {
        next_gesture = "other";
      }
      error_reporter->Report("# Hold the wand upright, you should see the left LED light up");
      error_reporter->Report("# Hold it steady for three seconds, and then perform the %s gesture", next_gesture);
      state = ePendingUpright; 
    } break;  
    case ePendingUpright: {
      if (is_upright) {
        upright_found_time = counter;
        state = eIsUpright;
      }
    } break;
    case eIsUpright: {
      if (is_upright) {
        digitalWrite(LED_BUILTIN, HIGH);
        if ((counter - upright_found_time) > 75) {
          digitalWrite(LED_BUILTIN, LOW);
          error_reporter->Report("# Start the %s gesture", next_gesture);
          gesture_start_time = counter;
          state = eRecordingGesture;         
        }
      } else {
        digitalWrite(LED_BUILTIN, LOW);
        state = ePendingUpright;
      }
    } break;
    case eRecordingGesture: {
      const int recording_time = 100;
      if ((counter - gesture_start_time) > recording_time) {
        error_reporter->Report("****************");
        error_reporter->Report("gesture: %s", next_gesture);
        for (int offset = recording_time; offset > 0; --offset) {
          const int array_offset = (input_length - (offset * 3));
          const int x = int(sensor_arena[array_offset + 0]); 
          const int y = int(sensor_arena[array_offset + 1]); 
          const int z = int(sensor_arena[array_offset + 2]); 
          error_reporter->Report("x: %d y:%d z:%d", x, y, z);
        }
        error_reporter->Report("~~~~~~~~~~~~~~~~");
        state = eStarting;
      }
    } break;
  }

  ++counter;
}
