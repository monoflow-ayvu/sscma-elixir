#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

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
std::atomic<int> g_frame_count(0);
std::chrono::steady_clock::time_point g_start_time;

// CLI configuration (read-only after init)
static std::string g_cli_model_path;
static bool g_cli_emit_base64 = true;
static bool g_cli_single_mode = false;
static float g_cli_threshold = 0.5f;
static bool g_cli_base64_payload = false;
static bool g_cli_quiet = false;
static std::string g_cli_publish_http_url;

// Camera/transmission settings (CLI configurable)
static int g_camera_width = 0;
static int g_camera_height = 0;
static int g_camera_fps = 3;

// Model info (read-only after init)
static std::string g_model_name_str;
static int g_model_type = 0;
static int g_model_input_type = 0;
static int g_model_output_type = 0;
static int g_model_input_width = 0;
static int g_model_input_height = 0;

// ===========================================================================
// Thread-Safe Shared State for HTTP Publishing
// ===========================================================================
struct HttpState {
  std::mutex mutex;

  // Frame data ready for HTTP publishing
  bool has_new_frame = false;
  std::string json_data;
  int frame_number = 0;
  
  // Track if at least one HTTP POST was successfully sent
  bool at_least_one_post_sent = false;
};

// ===========================================================================
// Image Preprocessing Function (similar to SDK example)
// ===========================================================================
::cv::Mat preprocessImage(::cv::Mat& image, ma::Model* model) {
  std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
  int ih = image.rows;
  int iw = image.cols;
  int oh = 0;
  int ow = 0;
  
  if (model->getInputType() == MA_INPUT_TYPE_IMAGE) {
    const ma_img_t* model_input = reinterpret_cast<const ma_img_t*>(model->getInput());
    oh = model_input->height;
    ow = model_input->width;
  }

  ::cv::Mat resizedImage;
  double resize_scale = std::min((double)oh / ih, (double)ow / iw);
  int nh = (int)(ih * resize_scale);
  int nw = (int)(iw * resize_scale);
  ::cv::resize(image, resizedImage, ::cv::Size(nw, nh));

  int top = (oh - nh) / 2;
  int bottom = (oh - nh) - top;
  int left = (ow - nw) / 2;
  int right = (ow - nw) - left;

  ::cv::Mat paddedImage;
  ::cv::copyMakeBorder(resizedImage, paddedImage, top, bottom, left, right, 
                       ::cv::BORDER_CONSTANT, ::cv::Scalar::all(0));

  ::cv::cvtColor(paddedImage, paddedImage, ::cv::COLOR_BGR2RGB);
  std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;
  MA_LOGD(TAG, "preprocessImage time: %f ms", duration.count() * 1000);
  return paddedImage;
}

// ===========================================================================
// Main Processing Pipeline
// Retrieves JPEG, decodes, preprocesses, runs inference, outputs JSON
// ===========================================================================
void mainProcessingPipeline(Camera *camera, ma::Model *model,
                            ma::engine::EngineCVI *engine,
                            HttpState *http_state) {
  MA_LOGI(TAG, "Main processing pipeline started");

  // Pre-allocate buffer for base64 encoding
  std::string base64_buffer;
  base64_buffer.reserve(g_camera_width * g_camera_height * 2);

  auto last_frame_time = std::chrono::steady_clock::now();

  while (g_running.load()) {
    // Retrieve RAW RGB frame from Channel 0 for TPU processing
    ma_img_t raw_frame;
    ma_err_t ret = camera->retrieveFrame(raw_frame, MA_PIXEL_FORMAT_RGB888);
    if (ret != MA_OK) {
      if (ret != MA_AGAIN) {
        MA_LOGE(TAG, "retrieveFrame (raw) failed: %d", ret);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    int frame_idx = g_frame_count.fetch_add(1) + 1;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now - g_start_time)
                       .count();
    auto elapsed_since_last_frame =
        std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                              last_frame_time)
            .count();

    MA_LOGI(TAG, "frame %d, elapsed since last frame %ld ms", frame_idx, elapsed_since_last_frame);

    try {
      auto tpu_start = std::chrono::steady_clock::now();
      auto step_start = tpu_start;

      // Retrieve JPEG frame from Channel 1 for transmission (non-blocking)
      ma_img_t jpeg_frame;
      std::string jpeg_data;
      uint32_t jpeg_size = 0;
      bool have_jpeg = false;
      
      if (g_cli_emit_base64) {
        ret = camera->retrieveFrame(jpeg_frame, MA_PIXEL_FORMAT_JPEG);
        if (ret == MA_OK) {
          jpeg_data.assign(reinterpret_cast<const char*>(jpeg_frame.data), jpeg_frame.size);
          jpeg_size = jpeg_frame.size;
          have_jpeg = true;
          camera->returnFrame(jpeg_frame);
        }
      }

      auto step_end = std::chrono::steady_clock::now();
      MA_LOGD(TAG, "[TIMING] JPEG retrieve: %.2f ms (have_jpeg=%d)", 
              std::chrono::duration<double, std::milli>(step_end - step_start).count(), have_jpeg);
      step_start = step_end;

      // Raw frame is already at model input dimensions, just need to set up ma_img_t
      // The raw frame is RGB888 format from the camera
      ma_img_t img;
      img.data = raw_frame.data;
      img.size = raw_frame.size;
      img.width = raw_frame.width;
      img.height = raw_frame.height;
      img.format = MA_PIXEL_FORMAT_RGB888;
      img.rotate = MA_PIXEL_ROTATE_0;
      img.physical = false;

      step_end = std::chrono::steady_clock::now();
      MA_LOGD(TAG, "[TIMING] Setup raw frame: %.2f ms", 
              std::chrono::duration<double, std::milli>(step_end - step_start).count());
      step_start = step_end;

      // Run inference based on model type - pass the image pointer
      ret = MA_OK;
      ma::model::Classifier *classifier = nullptr;
      ma::model::PoseDetector *pose = nullptr;
      ma::model::Segmentor *seg = nullptr;
      ma::model::Detector *det = nullptr;

      if (model->getOutputType() == MA_OUTPUT_TYPE_CLASS) {
        classifier = static_cast<ma::model::Classifier *>(model);
        classifier->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, g_cli_threshold);
        ret = classifier->run(&img);
      } else if (model->getOutputType() == MA_OUTPUT_TYPE_KEYPOINT) {
        pose = static_cast<ma::model::PoseDetector *>(model);
        pose->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, g_cli_threshold);
        ret = pose->run(&img);
      } else if (model->getOutputType() == MA_OUTPUT_TYPE_SEGMENT) {
        seg = static_cast<ma::model::Segmentor *>(model);
        seg->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, g_cli_threshold);
        ret = seg->run(&img);
      } else if (model->getOutputType() == MA_OUTPUT_TYPE_BBOX ||
                 model->getOutputType() == MA_OUTPUT_TYPE_TENSOR) {
        det = static_cast<ma::model::Detector *>(model);
        det->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, g_cli_threshold);
        ret = det->run(&img);
      } else {
        ret = MA_ENOTSUP;
      }

      step_end = std::chrono::steady_clock::now();
      MA_LOGD(TAG, "[TIMING] Model inference: %.2f ms", 
              std::chrono::duration<double, std::milli>(step_end - step_start).count());
      step_start = step_end;

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

        step_end = std::chrono::steady_clock::now();
        MA_LOGD(TAG, "[TIMING] Build TPU tensor metadata: %.2f ms", 
                std::chrono::duration<double, std::milli>(step_end - step_start).count());
        step_start = step_end;

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
            MA_LOGD(TAG, "classifier result: %s", cj.dump().c_str());
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
            MA_LOGD(TAG, "detector result: %s", dj.dump().c_str());
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
            MA_LOGD(TAG, "pose result: %s", pj.dump().c_str());
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
            MA_LOGD(TAG, "segment result: %s", sj.dump().c_str());
            results.push_back(std::move(sj));
          }
        }

        step_end = std::chrono::steady_clock::now();
        MA_LOGD(TAG, "[TIMING] Extract results: %.2f ms", 
                std::chrono::duration<double, std::milli>(step_end - step_start).count());
        step_start = step_end;

        // Build complete JSON output with JPEG and TPU results
        nlohmann::json j;
        j["frame_num"] = frame_idx;
        j["elapsed"] = elapsed;
        j["elapsed_since_last_frame"] = elapsed_since_last_frame;
        j["jpeg_size"] = jpeg_size;

        // Add base64-encoded JPEG if requested and available
        if (g_cli_emit_base64 && have_jpeg) {
          base64_buffer = macaron::Base64::Encode(jpeg_data);
          j["frame"] = base64_buffer;

          step_end = std::chrono::steady_clock::now();
          MA_LOGD(TAG, "[TIMING] Base64 encode JPEG: %.2f ms", 
                  std::chrono::duration<double, std::milli>(step_end - step_start).count());
          step_start = step_end;
        }

        // Add TPU results
        j["tpu"] = tpu_tensors;
        j["tpu_elapsed"] = tpu_elapsed;
        j["results"] = results;
        j["pre_shape"] = {{"w", pre_width}, {"h", pre_height}};
        j["camera_shape"] = {{"w", g_camera_width}, {"h", g_camera_height}};
        j["output_type"] = g_model_output_type;
        j["model_name"] = g_model_name_str;
        j["model_type"] = g_model_type;
        j["model_input_type"] = g_model_input_type;

        step_end = std::chrono::steady_clock::now();
        MA_LOGD(TAG, "[TIMING] Build JSON object: %.2f ms", 
                std::chrono::duration<double, std::milli>(step_end - step_start).count());
        step_start = step_end;

        // Prepare JSON string (only if needed for stdout or HTTP)
        std::string json_str;
        if (!g_cli_quiet || !g_cli_publish_http_url.empty()) {
          json_str = j.dump();

          step_end = std::chrono::steady_clock::now();
          MA_LOGD(TAG, "[TIMING] JSON serialize: %.2f ms", 
                  std::chrono::duration<double, std::milli>(step_end - step_start).count());
          step_start = step_end;
        }

        // Output to stdout (unless quiet mode)
        if (!g_cli_quiet) {
          if (g_cli_base64_payload) {
            std::string encoded = macaron::Base64::Encode(json_str);
            std::cout << encoded << std::endl;
          } else {
            std::cout << json_str << std::endl;
          }
          fflush(stdout);

          step_end = std::chrono::steady_clock::now();
          MA_LOGD(TAG, "[TIMING] Stdout output: %.2f ms", 
                  std::chrono::duration<double, std::milli>(step_end - step_start).count());
          step_start = step_end;
        }

        // Update HTTP state for async publishing (if URL is configured)
        if (!g_cli_publish_http_url.empty()) {
          std::lock_guard<std::mutex> lock(http_state->mutex);
          http_state->json_data = json_str;
          http_state->frame_number = frame_idx;
          http_state->has_new_frame = true;
        }

        // Log total pipeline time
        auto pipeline_end = std::chrono::steady_clock::now();
        MA_LOGI(TAG, "[TIMING] Total pipeline: %.2f ms", 
                std::chrono::duration<double, std::milli>(pipeline_end - tpu_start).count());

        // Exit in single mode after successful inference
        // If HTTP POST is enabled, wait until at least one POST is sent
        if (g_cli_single_mode) {
          if (!g_cli_publish_http_url.empty() && http_state != nullptr) {
            // Wait until at least one HTTP POST is successfully sent
            bool post_sent = false;
            while (!post_sent && g_running.load()) {
              std::this_thread::sleep_for(std::chrono::milliseconds(10));
              {
                std::lock_guard<std::mutex> lock(http_state->mutex);
                post_sent = http_state->at_least_one_post_sent;
              }
            }
          }
          g_running.store(false);
        }
      } else {
        // Inference failed
        MA_LOGE(TAG, "Inference failed with error: %d", ret);
      }

      last_frame_time = now;
    } catch (const std::exception& e) {
      MA_LOGE(TAG, "Processing pipeline exception: %s", e.what());
      last_frame_time = now;
    } catch (...) {
      MA_LOGE(TAG, "Processing pipeline unknown exception");
      last_frame_time = now;
    }

    // Return raw frame after processing
    camera->returnFrame(raw_frame);
  }

  MA_LOGI(TAG, "Main processing pipeline stopped");
}

// ===========================================================================
// HTTP Publishing Thread
// ===========================================================================
void httpPipeline(HttpState *http_state) {
  if (g_cli_publish_http_url.empty()) {
    return;  // No HTTP URL configured, thread exits immediately
  }

  MA_LOGI(TAG, "HTTP pipeline started");

  int last_sent_frame = 0;

  while (g_running.load()) {
    // Check for new frame data
    std::string json_to_send;
    int current_frame = 0;
    bool has_new = false;

    {
      std::lock_guard<std::mutex> lock(http_state->mutex);
      // Only send if frame is newer than what we last sent
      if (http_state->has_new_frame &&
          http_state->frame_number > last_sent_frame) {
        json_to_send = http_state->json_data;
        current_frame = http_state->frame_number;
        has_new = true;
        // Clear the flag after we've grabbed the data
        http_state->has_new_frame = false;
      }
    }

    if (has_new) {
      // Send HTTP request (outside of mutex lock to avoid blocking camera thread)
      http_headers headers;
      headers["Content-Type"] = "application/json";
      auto resp = requests::post(g_cli_publish_http_url.c_str(), json_to_send,
                                  headers);
      if (resp == NULL) {
        MA_LOGW(TAG, "HTTP POST to %s failed", g_cli_publish_http_url.c_str());
      } else if (resp->status_code >= 400) {
        MA_LOGW(TAG, "HTTP POST to %s returned status %d",
                g_cli_publish_http_url.c_str(), resp->status_code);
      } else {
        // Successfully sent, update last sent frame
        last_sent_frame = current_frame;
        // Mark that at least one POST was successfully sent
        {
          std::lock_guard<std::mutex> lock(http_state->mutex);
          http_state->at_least_one_post_sent = true;
        }
      }
    } else {
      // No new frame, sleep briefly to avoid busy-waiting
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  MA_LOGI(TAG, "HTTP pipeline stopped");
}


// ===========================================================================
// Main Entry Point
// ===========================================================================
int main(int argc, char **argv) {
  std::string modelPath;
  bool base64 = true;
  bool singleMode = false;
  float threshold = 0.5f;
  bool base64Payload = false;
  bool quiet = false;
  std::string publishHttpUrl;
  int cameraWidth = 0;   // 0 means use model input dimensions
  int cameraHeight = 0;  // 0 means use model input dimensions
  int cameraFps = 3;

  CLI::App app;
  app.add_option("-m,--model", modelPath, "Path to the model file")->required();
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
  app.add_flag("--quiet", quiet,
               "Disable stdout output (only logs will be printed)")
      ->default_val(false);
  app.add_option("--publish-http-to", publishHttpUrl,
                 "HTTP URL to POST JSON payload to")
      ->default_str("");
  app.add_option("--camera-width", cameraWidth,
                 "Width of the transmitted camera image (0 = use model input size)")
      ->default_val(0);
  app.add_option("--camera-height", cameraHeight,
                 "Height of the transmitted camera image (0 = use model input size)")
      ->default_val(0);
  app.add_option("--fps", cameraFps,
                 "FPS for camera capture / TPU processing")
      ->default_val(3);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  // Copy basic globals (camera dimensions will be set after model loading)
  g_cli_model_path = modelPath;
  g_cli_emit_base64 = base64;
  g_cli_single_mode = singleMode;
  g_cli_threshold = threshold;
  g_cli_base64_payload = base64Payload;
  g_cli_quiet = quiet;
  g_cli_publish_http_url = publishHttpUrl;
  g_camera_fps = cameraFps;

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

  // Get model input dimensions (used for TPU preprocessing)
  const ma_img_t *model_input =
      static_cast<const ma_img_t *>(model->getInput());
  g_model_input_width = model_input->width;
  g_model_input_height = model_input->height;
  MA_LOGI(TAG, "model input size: %dx%d", g_model_input_width, g_model_input_height);

  // Set camera dimensions (use model input size if not specified)
  g_camera_width = (cameraWidth > 0) ? cameraWidth : g_model_input_width;
  g_camera_height = (cameraHeight > 0) ? cameraHeight : g_model_input_height;
  MA_LOGI(TAG, "camera output size: %dx%d @ %d fps", g_camera_width, g_camera_height, g_camera_fps);

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

      // Configure RAW channel (0) - for TPU processing at model input dimensions
      value.i32 = 0;
      ret = camera->commandCtrl(Camera::CtrlType::kChannel,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kChannel 0 failed");
        return 1;
      }
      value.u16s[0] = g_model_input_width;
      value.u16s[1] = g_model_input_height;
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
      // RAW channel fps (same as camera fps)
      value.i32 = g_camera_fps;
      ret = camera->commandCtrl(Camera::CtrlType::kFps,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kFps (ch0) failed");
        return 1;
      }

      // Configure JPEG channel (1) - for transmission at camera dimensions
      value.i32 = 1;
      ret = camera->commandCtrl(Camera::CtrlType::kChannel,
                                Camera::CtrlMode::kWrite, value);
      if (ret != MA_OK) {
        MA_LOGE(TAG, "commandCtrl kChannel 1 failed");
        return 1;
      }
      value.u16s[0] = g_camera_width;
      value.u16s[1] = g_camera_height;
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
      // JPEG channel fps
      value.i32 = g_camera_fps;
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

  // Create shared state structure for HTTP publishing
  HttpState http_state;

  MA_LOGI(TAG, "Starting processing pipeline (Camera: %dx%d @ %d fps, Model: %dx%d)",
          g_camera_width, g_camera_height, g_camera_fps, g_model_input_width, g_model_input_height);

  // Start pipeline threads
  std::thread processing_thread(mainProcessingPipeline, camera, model, engine, &http_state);
  std::thread http_thread(httpPipeline, &http_state);

  // Wait for threads to complete
  processing_thread.join();
  http_thread.join();

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
