#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

#include <sscma.h>
#include <video.h>

#include "CLI11.hpp"
#include "base64.hpp"
#include "json.hpp"

using namespace ma;

#define TAG "model_detector"

// Global variables for frame capture control
std::atomic<bool> g_running(true);
std::atomic<int> g_frame_count(0);
std::chrono::steady_clock::time_point g_start_time;
std::chrono::steady_clock::time_point last_frame_time;
std::chrono::steady_clock::time_point g_last_tpu_time =
    std::chrono::steady_clock::time_point::min();

// CLI configuration
static std::string g_cli_model_path;
static int g_cli_tpu_delay_ms = 0;
static bool g_cli_emit_base64 = true;
static bool g_cli_single_mode = false;
static float g_cli_threshold = 0.5f;

int main(int argc, char **argv) {
  std::string modelPath;
  int tpuDelay = 0;
  bool base64 = true;
  bool singleMode = false;
  float threshold = 0.5f;

  CLI::App app;
  app.add_option("-m,--model", modelPath, "Path to the model file")->required();
  app.add_option("-t,--tpu-delay", tpuDelay,
                 "How fast to process frames in the TPU (in milliseconds)")
      ->default_str("0");
  app.add_option("-b,--base64", base64,
                 "Output frames as base64. If false, base64 is omitted.")
      ->default_str("true");
  app.add_option("-s,--single", singleMode,
                 "Exit after one successful frame with TPU data")
      ->default_str("false");
  app.add_option("--threshold", threshold, "Detection threshold")
      ->default_str("0.5");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  // Copy to globals for use
  g_cli_model_path = modelPath;
  g_cli_tpu_delay_ms = tpuDelay;
  g_cli_emit_base64 = base64;
  g_cli_single_mode = singleMode;
  g_cli_threshold = threshold;

  // Validate model path exists
  {
    std::ifstream f(g_cli_model_path);
    if (!f.good()) {
      std::cerr << "Model file not found: " << g_cli_model_path << std::endl;
      return -1;
    }
  }

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

  // Get model input dimensions
  const ma_img_t *model_input =
      static_cast<const ma_img_t *>(model->getInput());
  int input_width = model_input->width;
  int input_height = model_input->height;
  MA_LOGI(TAG, "model input size: %dx%d", input_width, input_height);

  // Initialize device and camera
  Device *device = Device::getInstance();
  Camera *camera = nullptr;

  Signal::install(
      {SIGINT, SIGSEGV, SIGABRT, SIGTRAP, SIGTERM, SIGHUP, SIGQUIT, SIGPIPE},
      [device](int sig) {
        std::cout << "Caught signal " << sig << std::endl;
        g_running.store(false);
        for (auto &sensor : device->getSensors()) {
          sensor->deInit();
        }
        exit(0);
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

      // Configure RAW channel (0) for RGB888 inference and JPEG encoding
      value.i32 = 0;
      ret = camera->commandCtrl(Camera::CtrlType::kChannel,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kChannel failed");
        return 1;
      }
      value.u16s[0] = input_width;
      value.u16s[1] = input_height;
      ret = camera->commandCtrl(Camera::CtrlType::kWindow,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kWindow failed");
        return 1;
      }
      value.i32 = static_cast<int>(MA_PIXEL_FORMAT_RGB888);
      ret = camera->commandCtrl(Camera::CtrlType::kFormat,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kFormat failed");
        return 1;
      }

      // value.i32 = 1;
      // camera->commandCtrl(Camera::CtrlType::kPhysical,
      // Camera::CtrlMode::kWrite, value);
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
  last_frame_time = g_start_time;

  // Pre-allocate reusable buffers outside the loop to eliminate per-frame
  // allocations
  std::vector<uint8_t> frame_buffer;
  frame_buffer.reserve(input_width * input_height * 3); // Reserve for RGB888
  std::vector<uint8_t> jpeg_buffer;
  jpeg_buffer.reserve(input_width *
                      input_height); // Typical JPEG is smaller than raw
  std::string base64_buffer;
  base64_buffer.reserve(input_width * input_height *
                        2); // Base64 is ~1.33x JPEG size

  // Pre-allocate OpenCV mats for reuse
  ::cv::Mat bgr_mat;

  // Compression params are constant, create once
  const std::vector<int> compression_params = {::cv::IMWRITE_JPEG_QUALITY, 90};

  // Cache model name to avoid repeated string copies
  const char *model_name_ptr = model->getName();
  const std::string model_name_str =
      model_name_ptr ? std::string(model_name_ptr) : "";

  while (g_running.load()) {
    ma_img_t frame;
    if (camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) == MA_OK) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now - g_start_time)
                         .count();
      auto elapsed_since_last_frame =
          std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                                last_frame_time)
              .count();
      int idx = g_frame_count.fetch_add(1) + 1;

      // Copy frame data for JPEG encoding (reuse buffer)
      size_t frame_data_size =
          frame.width * frame.height * 3; // RGB888 = 3 bytes per pixel
      frame_buffer.resize(frame_data_size);
      memcpy(frame_buffer.data(), frame.data, frame_data_size);

      // Convert RGB frame to JPEG using OpenCV (reuse bgr_mat)
      ::cv::Mat rgb_mat(frame.height, frame.width, CV_8UC3,
                        frame_buffer.data());
      ::cv::cvtColor(rgb_mat, bgr_mat, ::cv::COLOR_RGB2BGR);

      // Clear jpeg_buffer but keep capacity
      jpeg_buffer.clear();
      bool jpeg_success =
          ::cv::imencode(".jpg", bgr_mat, jpeg_buffer, compression_params);

      // Encode to base64 directly from jpeg_buffer without intermediate string
      // copy
      base64_buffer.clear();
      if (jpeg_success && g_cli_emit_base64) {
        base64_buffer = macaron::Base64::Encode(
            std::string(reinterpret_cast<const char *>(jpeg_buffer.data()),
                        jpeg_buffer.size()));
      }

      // Build JSON output
      nlohmann::json j;
      j["frame_num"] = idx;
      j["elapsed"] = elapsed;
      j["elapsed_since_last_frame"] = elapsed_since_last_frame;
      j["jpeg_size"] =
          jpeg_success ? static_cast<uint32_t>(jpeg_buffer.size()) : 0;
      if (g_cli_emit_base64 && jpeg_success) {
        j["frame"] = base64_buffer;
      }

      // Check if we should run TPU inference
      bool do_tpu = true;
      if (g_cli_tpu_delay_ms > 0 &&
          g_last_tpu_time != std::chrono::steady_clock::time_point::min()) {
        auto since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now - g_last_tpu_time)
                              .count();
        if (since_last < g_cli_tpu_delay_ms) {
          do_tpu = false;
        }
      }

      if (do_tpu && model && model->getInputType() == MA_INPUT_TYPE_IMAGE) {
        try {
          auto tpu_start = std::chrono::steady_clock::now();

          ma_tensor_t tensor = {
              .size = frame.size,
              .is_physical = true, // assume physical address
              .is_variable = false,
          };
          tensor.data.data = reinterpret_cast<void *>(frame.data);

          engine->setInput(0, tensor);
          model->setPreprocessDone(
              [camera, &frame](void *ctx) { camera->returnFrame(frame); });

          ma_err_t ret = MA_OK;
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

            // Get TPU output tensor metadata
            int outs = engine->getOutputSize();
            nlohmann::json tout = nlohmann::json::array();
            if (outs > 0) {
              tout.get_ptr<nlohmann::json::array_t *>()->reserve(outs);
            }
            for (int i = 0; i < outs; ++i) {
              auto t = engine->getOutput(i);
              nlohmann::json tj;
              tj["index"] = i;
              tj["size"] = static_cast<uint64_t>(t.size);
              tj["type"] = static_cast<int>(t.type);
              nlohmann::json shape = nlohmann::json::array();
              if (t.shape.size > 0) {
                shape.get_ptr<nlohmann::json::array_t *>()->reserve(
                    t.shape.size);
              }
              for (uint32_t d = 0; d < t.shape.size; ++d) {
                shape.push_back(t.shape.dims[d]);
              }
              tj["shape"] = shape;
              tj["quant_scale"] = t.quant_param.scale;
              tj["quant_zero"] = t.quant_param.zero_point;
              tj["is_physical"] = t.is_physical;
              tj["is_variable"] = t.is_variable;
              // Avoid string copy for tensor name
              if (t.name) {
                tj["name"] = t.name;
              } else {
                tj["name"] = "";
              }
              tout.push_back(tj);
            }
            j["tpu"] = tout;
            j["tpu_elapsed"] = tpu_elapsed;
            g_last_tpu_time = now;
            j["output_type"] = model->getOutputType();
            j["model_name"] = model_name_str;
            j["model_type"] = model->getType();
            j["model_input_type"] = model->getInputType();

            // Extract and format results based on model type
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
                results.push_back(cj);
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
                // absolute coords in preprocessed image space
                float x1 = (it.x - it.w / 2.0f) * pre_width;
                float y1 = (it.y - it.h / 2.0f) * pre_height;
                float x2 = (it.x + it.w / 2.0f) * pre_width;
                float y2 = (it.y + it.h / 2.0f) * pre_height;
                dj["abs"] = {{"x1", x1}, {"y1", y1}, {"x2", x2}, {"y2", y2}};
                results.push_back(dj);
              }
            } else if (pose) {
              const auto &r = pose->getResults();
              for (const auto &it : r) {
                nlohmann::json pj;
                pj["kind"] = "pose";
                pj["box"] = {
                    {"x", it.box.x},         {"y", it.box.y},
                    {"w", it.box.w},         {"h", it.box.h},
                    {"score", it.box.score}, {"target", it.box.target}};
                float x1 = (it.box.x - it.box.w / 2.0f) * pre_width;
                float y1 = (it.box.y - it.box.h / 2.0f) * pre_height;
                float x2 = (it.box.x + it.box.w / 2.0f) * pre_width;
                float y2 = (it.box.y + it.box.h / 2.0f) * pre_height;
                pj["box_abs"] = {
                    {"x1", x1}, {"y1", y1}, {"x2", x2}, {"y2", y2}};
                nlohmann::json pts = nlohmann::json::array();
                for (const auto &pt : it.pts) {
                  pts.push_back({{"x", pt.x}, {"y", pt.y}, {"z", pt.z}});
                }
                pj["keypoints"] = pts;
                results.push_back(pj);
              }
            } else if (seg) {
              const auto &r = seg->getResults();
              for (const auto &it : r) {
                nlohmann::json sj;
                sj["kind"] = "segment";
                sj["box"] = {
                    {"x", it.box.x},         {"y", it.box.y},
                    {"w", it.box.w},         {"h", it.box.h},
                    {"score", it.box.score}, {"target", it.box.target}};
                float x1 = (it.box.x - it.box.w / 2.0f) * pre_width;
                float y1 = (it.box.y - it.box.h / 2.0f) * pre_height;
                float x2 = (it.box.x + it.box.w / 2.0f) * pre_width;
                float y2 = (it.box.y + it.box.h / 2.0f) * pre_height;
                sj["box_abs"] = {
                    {"x1", x1}, {"y1", y1}, {"x2", x2}, {"y2", y2}};
                sj["mask"] = {{"width", it.mask.width},
                              {"height", it.mask.height}};
                // Note: mask.data is a pointer, we don't serialize it directly
                results.push_back(sj);
              }
            }
            j["results"] = results;
            j["pre_shape"] = {{"w", pre_width}, {"h", pre_height}};

            // Exit after one successful frame in single mode
            if (g_cli_single_mode) {
              g_running.store(false);
            }
          } else {
            j["tpu_err"] = ret;
            camera->returnFrame(frame);
          }
        } catch (...) {
          j["tpu_err"] = -1;
          camera->returnFrame(frame);
        }
      } else {
        // TPU skipped due to delay or other reason
        camera->returnFrame(frame);
      }

      std::cout << j.dump() << std::endl;
      last_frame_time = now;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  camera->stopStream();
  if (model) {
    ma::ModelFactory::remove(model);
  }
  if (engine) {
    delete engine;
  }

  return 0;
}
