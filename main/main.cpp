#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include <sscma.h>
#include <video.h>

#include <hv/requests.h>

#include "CLI11.hpp"
#include "base64.hpp"
#include "json.hpp"

using namespace ma;

#define TAG "model_detector"

// Global variables for control
std::atomic<bool> g_running(true);
std::atomic<int> g_camera_frame_count(0);
std::atomic<int> g_tpu_frame_count(0);
std::chrono::steady_clock::time_point g_start_time;

// CLI configuration (read-only after init)
static std::string g_cli_model_path;
static int g_cli_tpu_delay_ms = 0;
static int g_cli_camera_fps = 30;
static bool g_cli_emit_base64 = true;
static bool g_cli_single_mode = false;
static float g_cli_threshold = 0.5f;
static bool g_cli_base64_payload = false;
static std::string g_cli_publish_http_url;

// Model info (read-only after init)
static std::string g_model_name_str;
static int g_model_type = 0;
static int g_model_input_type = 0;
static int g_model_output_type = 0;
static int g_input_width = 0;
static int g_input_height = 0;

// ===========================================================================
// Thread-Safe Shared State for TPU Results
// ===========================================================================
struct TPUState {
  std::mutex mutex;

  // TPU results
  bool has_results = false;
  int64_t tpu_elapsed_ms = 0;
  nlohmann::json tpu_tensors;
  nlohmann::json results;
  int pre_width = 0;
  int pre_height = 0;
  int tpu_frame_number = 0;

  // Error state
  ma_err_t last_error = MA_OK;
};

// ===========================================================================
// TPU Pipeline Thread
// ===========================================================================
void tpuPipeline(Camera *camera, ma::Model *model,
                 ma::engine::EngineCVI *engine, TPUState *tpu_state) {
  MA_LOGI(TAG, "TPU pipeline started");

  auto last_inference_time = std::chrono::steady_clock::now();

  while (g_running.load()) {
    // Respect tpu_delay_ms timing between inferences
    if (g_cli_tpu_delay_ms > 0) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now - last_inference_time)
                         .count();

      if (elapsed < g_cli_tpu_delay_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
    }

    // Retrieve RGB frame from Channel 0
    ma_img_t frame;
    ma_err_t ret = camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888);
    if (ret != MA_OK) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    int frame_idx = g_tpu_frame_count.fetch_add(1) + 1;
    last_inference_time = std::chrono::steady_clock::now();

    try {
      auto tpu_start = std::chrono::steady_clock::now();

      // Set up tensor with physical address
      ma_tensor_t tensor = {
          .size = frame.size,
          .is_physical = true,
          .is_variable = false,
      };
      tensor.data.data = reinterpret_cast<void *>(frame.data);

      engine->setInput(0, tensor);

      // Return frame after preprocessing is done
      model->setPreprocessDone(
          [camera, frame](void *ctx) mutable { camera->returnFrame(frame); });

      // Run inference based on model type
      ret = MA_OK;
      ma::model::Classifier *classifier = nullptr;
      ma::model::PoseDetector *pose = nullptr;
      ma::model::Segmentor *seg = nullptr;
      ma::model::Detector *det = nullptr;

      if (model->getOutputType() == MA_OUTPUT_TYPE_CLASS) {
        classifier = static_cast<ma::model::Classifier *>(model);
        classifier->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, g_cli_threshold);
        ret = classifier->run(nullptr);
      } else if (model->getOutputType() == MA_OUTPUT_TYPE_KEYPOINT) {
        pose = static_cast<ma::model::PoseDetector *>(model);
        pose->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, g_cli_threshold);
        ret = pose->run(nullptr);
      } else if (model->getOutputType() == MA_OUTPUT_TYPE_SEGMENT) {
        seg = static_cast<ma::model::Segmentor *>(model);
        seg->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, g_cli_threshold);
        ret = seg->run(nullptr);
      } else if (model->getOutputType() == MA_OUTPUT_TYPE_BBOX ||
                 model->getOutputType() == MA_OUTPUT_TYPE_TENSOR) {
        det = static_cast<ma::model::Detector *>(model);
        det->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, g_cli_threshold);
        ret = det->run(nullptr);
      } else {
        ret = MA_ENOTSUP;
        camera->returnFrame(frame);
      }

      if (ret == MA_OK) {
        auto tpu_end = std::chrono::steady_clock::now();
        auto tpu_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(tpu_end -
                                                                  tpu_start)
                .count();

        // Build TPU tensor metadata
        int outs = engine->getOutputSize();
        nlohmann::json tpu_tensors = nlohmann::json::array();
        if (outs > 0) {
          tpu_tensors.get_ptr<nlohmann::json::array_t *>()->reserve(outs);
        }
        for (int i = 0; i < outs; ++i) {
          auto t = engine->getOutput(i);
          nlohmann::json tj;
          tj["index"] = i;
          tj["size"] = static_cast<uint64_t>(t.size);
          tj["type"] = static_cast<int>(t.type);
          nlohmann::json shape = nlohmann::json::array();
          if (t.shape.size > 0) {
            shape.get_ptr<nlohmann::json::array_t *>()->reserve(t.shape.size);
          }
          for (uint32_t d = 0; d < t.shape.size; ++d) {
            shape.push_back(t.shape.dims[d]);
          }
          tj["shape"] = shape;
          tj["quant_scale"] = t.quant_param.scale;
          tj["quant_zero"] = t.quant_param.zero_point;
          tj["is_physical"] = t.is_physical;
          tj["is_variable"] = t.is_variable;
          tj["name"] = t.name ? t.name : "";
          tpu_tensors.push_back(std::move(tj));
        }

        // Extract results based on model type
        nlohmann::json results = nlohmann::json::array();
        const ma_img_t *model_input =
            static_cast<const ma_img_t *>(model->getInput());
        int pre_width = model_input->width;
        int pre_height = model_input->height;

        if (classifier) {
          const auto &r = classifier->getResults();
          for (const auto &it : r) {
            nlohmann::json cj;
            cj["kind"] = "classifier";
            cj["score"] = it.score;
            cj["target"] = it.target;
            results.push_back(std::move(cj));
          }
        } else if (det) {
          const auto &r = det->getResults();
          for (const auto &it : r) {
            nlohmann::json dj;
            dj["kind"] = "detector";
            dj["x"] = it.x;
            dj["y"] = it.y;
            dj["w"] = it.w;
            dj["h"] = it.h;
            dj["score"] = it.score;
            dj["target"] = it.target;
            float x1 = (it.x - it.w / 2.0f) * pre_width;
            float y1 = (it.y - it.h / 2.0f) * pre_height;
            float x2 = (it.x + it.w / 2.0f) * pre_width;
            float y2 = (it.y + it.h / 2.0f) * pre_height;
            dj["abs"] = {{"x1", x1}, {"y1", y1}, {"x2", x2}, {"y2", y2}};
            results.push_back(std::move(dj));
          }
        } else if (pose) {
          const auto &r = pose->getResults();
          for (const auto &it : r) {
            nlohmann::json pj;
            pj["kind"] = "pose";
            pj["box"] = {{"x", it.box.x},         {"y", it.box.y},
                         {"w", it.box.w},         {"h", it.box.h},
                         {"score", it.box.score}, {"target", it.box.target}};
            float x1 = (it.box.x - it.box.w / 2.0f) * pre_width;
            float y1 = (it.box.y - it.box.h / 2.0f) * pre_height;
            float x2 = (it.box.x + it.box.w / 2.0f) * pre_width;
            float y2 = (it.box.y + it.box.h / 2.0f) * pre_height;
            pj["box_abs"] = {{"x1", x1}, {"y1", y1}, {"x2", x2}, {"y2", y2}};
            nlohmann::json pts = nlohmann::json::array();
            for (const auto &pt : it.pts) {
              pts.push_back({{"x", pt.x}, {"y", pt.y}, {"z", pt.z}});
            }
            pj["keypoints"] = std::move(pts);
            results.push_back(std::move(pj));
          }
        } else if (seg) {
          const auto &r = seg->getResults();
          for (const auto &it : r) {
            nlohmann::json sj;
            sj["kind"] = "segment";
            sj["box"] = {{"x", it.box.x},         {"y", it.box.y},
                         {"w", it.box.w},         {"h", it.box.h},
                         {"score", it.box.score}, {"target", it.box.target}};
            float x1 = (it.box.x - it.box.w / 2.0f) * pre_width;
            float y1 = (it.box.y - it.box.h / 2.0f) * pre_height;
            float x2 = (it.box.x + it.box.w / 2.0f) * pre_width;
            float y2 = (it.box.y + it.box.h / 2.0f) * pre_height;
            sj["box_abs"] = {{"x1", x1}, {"y1", y1}, {"x2", x2}, {"y2", y2}};
            sj["mask"] = {{"width", it.mask.width}, {"height", it.mask.height}};
            results.push_back(std::move(sj));
          }
        }

        // Update shared state with mutex protection
        {
          std::lock_guard<std::mutex> lock(tpu_state->mutex);
          tpu_state->has_results = true;
          tpu_state->tpu_elapsed_ms = tpu_elapsed;
          tpu_state->tpu_tensors = std::move(tpu_tensors);
          tpu_state->results = std::move(results);
          tpu_state->pre_width = pre_width;
          tpu_state->pre_height = pre_height;
          tpu_state->tpu_frame_number = frame_idx;
          tpu_state->last_error = MA_OK;
        }

        // Exit in single mode after successful inference
        if (g_cli_single_mode) {
          g_running.store(false);
        }
      } else {
        // Inference failed
        std::lock_guard<std::mutex> lock(tpu_state->mutex);
        tpu_state->last_error = ret;
        camera->returnFrame(frame);
      }
    } catch (...) {
      MA_LOGE(TAG, "TPU pipeline exception");
      std::lock_guard<std::mutex> lock(tpu_state->mutex);
      tpu_state->last_error = MA_FAILED;
      camera->returnFrame(frame);
    }
  }

  MA_LOGI(TAG, "TPU pipeline stopped");
}

// ===========================================================================
// Camera Pipeline Thread
// ===========================================================================
void cameraPipeline(Camera *camera, TPUState *tpu_state) {
  MA_LOGI(TAG, "Camera pipeline started");

  // Pre-allocate buffer for base64 encoding
  std::string base64_buffer;
  base64_buffer.reserve(g_input_width * g_input_height * 2);

  auto last_frame_time = std::chrono::steady_clock::now();

  while (g_running.load()) {
    // Retrieve JPEG frame from Channel 1
    ma_img_t jpeg_frame;
    ma_err_t ret = camera->retrieveFrame(jpeg_frame, MA_PIXEL_FORMAT_JPEG);
    if (ret != MA_OK) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now - g_start_time)
                       .count();
    auto elapsed_since_last_frame =
        std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                              last_frame_time)
            .count();
    int frame_idx = g_camera_frame_count.fetch_add(1) + 1;

    // Encode JPEG to base64
    base64_buffer.clear();
    if (g_cli_emit_base64) {
      base64_buffer = macaron::Base64::Encode(std::string(
          reinterpret_cast<const char *>(jpeg_frame.data), jpeg_frame.size));
    }

    // Build JSON output
    nlohmann::json j;
    j["frame_num"] = frame_idx;
    j["elapsed"] = elapsed;
    j["elapsed_since_last_frame"] = elapsed_since_last_frame;
    j["jpeg_size"] = static_cast<uint32_t>(jpeg_frame.size);
    if (g_cli_emit_base64) {
      j["frame"] = base64_buffer;
    }

    // Copy TPU state under mutex (quick copy, then release)
    {
      std::lock_guard<std::mutex> lock(tpu_state->mutex);
      if (tpu_state->has_results) {
        j["tpu"] = tpu_state->tpu_tensors;
        j["tpu_elapsed"] = tpu_state->tpu_elapsed_ms;
        j["results"] = tpu_state->results;
        j["pre_shape"] = {{"w", tpu_state->pre_width},
                          {"h", tpu_state->pre_height}};
        j["tpu_frame_num"] = tpu_state->tpu_frame_number;
        j["output_type"] = g_model_output_type;
        j["model_name"] = g_model_name_str;
        j["model_type"] = g_model_type;
        j["model_input_type"] = g_model_input_type;
      }
      if (tpu_state->last_error != MA_OK) {
        j["tpu_err"] = tpu_state->last_error;
      }
    }

    // Return JPEG frame immediately after reading data
    camera->returnFrame(jpeg_frame);

    // Prepare JSON string
    std::string json_str = j.dump();

    // Output to stdout
    if (g_cli_base64_payload) {
      std::string encoded = macaron::Base64::Encode(json_str);
      std::cout << encoded << std::endl;
    } else {
      std::cout << json_str << std::endl;
    }

    // Publish to HTTP if URL is provided
    if (!g_cli_publish_http_url.empty()) {
      http_headers headers;
      headers["Content-Type"] = "application/json";
      auto resp =
          requests::post(g_cli_publish_http_url.c_str(), json_str, headers);
      if (resp == NULL) {
        MA_LOGW(TAG, "HTTP POST to %s failed", g_cli_publish_http_url.c_str());
      } else if (resp->status_code >= 400) {
        MA_LOGW(TAG, "HTTP POST to %s returned status %d",
                g_cli_publish_http_url.c_str(), resp->status_code);
      }
    }

    last_frame_time = now;
  }

  MA_LOGI(TAG, "Camera pipeline stopped");
}

// ===========================================================================
// Main Entry Point
// ===========================================================================
int main(int argc, char **argv) {
  std::string modelPath;
  int tpuDelay = 0;
  int cameraFps = 30;
  bool base64 = true;
  bool singleMode = false;
  float threshold = 0.5f;
  bool base64Payload = false;
  std::string publishHttpUrl;

  CLI::App app;
  app.add_option("-m,--model", modelPath, "Path to the model file")->required();
  app.add_option("-t,--tpu-delay", tpuDelay,
                 "How fast to process frames in the TPU (in milliseconds)")
      ->default_str("0");
  app.add_option("-f,--fps", cameraFps, "Camera output FPS (frames per second)")
      ->default_str("30");
  app.add_option("-b,--base64", base64,
                 "Output frames as base64. If false, base64 is omitted.")
      ->default_str("true");
  app.add_option("-s,--single", singleMode,
                 "Exit after one successful frame with TPU data")
      ->default_str("false");
  app.add_option("--threshold", threshold, "Detection threshold")
      ->default_str("0.5");
  app.add_option("--base64-payload", base64Payload,
                 "Encode entire JSON payload as base64")
      ->default_str("false");
  app.add_option("--publish-http-to", publishHttpUrl,
                 "HTTP URL to POST JSON payload to")
      ->default_str("");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  // Copy to globals for use by pipeline threads
  g_cli_model_path = modelPath;
  g_cli_tpu_delay_ms = tpuDelay;
  g_cli_camera_fps = cameraFps;
  g_cli_emit_base64 = base64;
  g_cli_single_mode = singleMode;
  g_cli_threshold = threshold;
  g_cli_base64_payload = base64Payload;
  g_cli_publish_http_url = publishHttpUrl;

  // Validate model path exists
  {
    std::ifstream f(g_cli_model_path);
    if (!f.good()) {
      std::cerr << "Model file not found: " << g_cli_model_path << std::endl;
      return -1;
    }
  }

  // Initialize engine and load model
  ma_err_t ret = MA_OK;
  auto *engine = new ma::engine::EngineCVI();
  ret = engine->init();
  if (ret != MA_OK) {
    MA_LOGE(TAG, "engine init failed");
    return 1;
  }
  ret = engine->load(g_cli_model_path.c_str());
  MA_LOGI(TAG, "engine load model %s", g_cli_model_path.c_str());
  if (ret != MA_OK) {
    MA_LOGE(TAG, "engine load model failed");
    return 1;
  }

  ma::Model *model = ma::ModelFactory::create(engine);
  if (model == nullptr) {
    MA_LOGE(TAG, "model not supported");
    return 1;
  }

  MA_LOGI(TAG, "model type: %d", model->getType());
  MA_LOGI(TAG, "threshold: %f", g_cli_threshold);

  if (model->getInputType() != MA_INPUT_TYPE_IMAGE) {
    MA_LOGE(TAG, "model input type not supported");
    return 1;
  }

  // Cache model info for pipeline threads
  const char *model_name_ptr = model->getName();
  g_model_name_str = model_name_ptr ? std::string(model_name_ptr) : "";
  g_model_type = model->getType();
  g_model_input_type = model->getInputType();
  g_model_output_type = model->getOutputType();

  // Get model input dimensions
  const ma_img_t *model_input =
      static_cast<const ma_img_t *>(model->getInput());
  g_input_width = model_input->width;
  g_input_height = model_input->height;
  MA_LOGI(TAG, "model input size: %dx%d", g_input_width, g_input_height);

  // Initialize device and camera
  Device *device = Device::getInstance();
  Camera *camera = nullptr;

  Signal::install(
      {SIGINT, SIGSEGV, SIGABRT, SIGTRAP, SIGTERM, SIGHUP, SIGQUIT, SIGPIPE},
      [device](int sig) {
        MA_LOGI(TAG, "Caught signal %d, shutting down...", sig);
        g_running.store(false);
      });

  ret = MA_OK;
  Camera::CtrlValue value;
  for (auto &sensor : device->getSensors()) {
    if (sensor->getType() == ma::Sensor::Type::kCamera) {
      camera = static_cast<Camera *>(sensor);
      ret = camera->init(0);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "camera init failed");
        return 1;
      }

      // Configure RAW channel (0) for RGB888 - used for TPU inference
      value.i32 = 0;
      ret = camera->commandCtrl(Camera::CtrlType::kChannel,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kChannel 0 failed");
        return 1;
      }
      value.u16s[0] = g_input_width;
      value.u16s[1] = g_input_height;
      ret = camera->commandCtrl(Camera::CtrlType::kWindow,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kWindow (ch0) failed");
        return 1;
      }
      value.i32 = static_cast<int>(MA_PIXEL_FORMAT_RGB888);
      ret = camera->commandCtrl(Camera::CtrlType::kFormat,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kFormat (ch0) failed");
        return 1;
      }
      // TPU channel fps based on tpu_delay_ms
      if (g_cli_tpu_delay_ms > 0) {
        value.i32 = std::max(
            1, static_cast<int>(std::round(1000.0f / g_cli_tpu_delay_ms)));
      } else {
        value.i32 = 30;
      }
      ret = camera->commandCtrl(Camera::CtrlType::kFps,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kFps (ch0) failed");
        return 1;
      }

      // Configure JPEG channel (1) for hardware-encoded JPEG output
      value.i32 = 1;
      ret = camera->commandCtrl(Camera::CtrlType::kChannel,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kChannel 1 failed");
        return 1;
      }
      value.u16s[0] = g_input_width;
      value.u16s[1] = g_input_height;
      ret = camera->commandCtrl(Camera::CtrlType::kWindow,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kWindow (ch1) failed");
        return 1;
      }
      value.i32 = static_cast<int>(MA_PIXEL_FORMAT_JPEG);
      ret = camera->commandCtrl(Camera::CtrlType::kFormat,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kFormat (ch1) failed");
        return 1;
      }
      // Camera output fps
      value.i32 = g_cli_camera_fps;
      ret = camera->commandCtrl(Camera::CtrlType::kFps,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kFps (ch1) failed");
        return 1;
      }
      break;
    }
  }

  if (!camera) {
    MA_LOGE(TAG, "No camera found");
    return 1;
  }

  ret = camera->startStream(Camera::StreamMode::kRefreshOnReturn);
  if (ret != MA_OK) {
    MA_LOGE(TAG, "camera startStream failed");
    return 1;
  }

  g_start_time = std::chrono::steady_clock::now();

  // Create shared TPU state
  TPUState tpu_state;

  MA_LOGI(TAG, "Starting dual pipeline (TPU delay: %d ms, Camera FPS: %d)",
          g_cli_tpu_delay_ms, g_cli_camera_fps);

  // Start pipeline threads
  std::thread tpu_thread(tpuPipeline, camera, model, engine, &tpu_state);
  std::thread camera_thread(cameraPipeline, camera, &tpu_state);

  // Wait for threads to complete
  tpu_thread.join();
  camera_thread.join();

  MA_LOGI(TAG, "Pipelines stopped, cleaning up...");

  // Cleanup
  camera->stopStream();
  if (model) {
    ma::ModelFactory::remove(model);
  }
  if (engine) {
    delete engine;
  }

  // Deinitialize sensors
  for (auto &sensor : device->getSensors()) {
    sensor->deInit();
  }

  MA_LOGI(TAG, "Shutdown complete");
  return 0;
}
