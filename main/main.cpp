#include <chrono>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <signal.h>
#include <string>
#include <atomic>
#include <unistd.h>
#include <string.h>
#include <thread>

#include <opencv2/opencv.hpp>

#include <sscma.h>

#include "base64.hpp"
#include "json.hpp"
#include "CLI11.hpp"

extern "C"
{
#include <video.h>
}

extern "C"
{
// CVITEK
#include <cvi_vb.h>
#include <cvi_sys.h>
#include <cvi_vi.h>
#include <cvi_comm_vi.h>
#include <cvi_vpss.h>
#include <cvi_venc.h>
#include <cvi_comm_venc.h>
// ISP / AE / AWB / MIPI
#include <cvi_isp.h>
#include <cvi_comm_isp.h>
#include <cvi_ae.h>
#include <cvi_awb.h>
#include <cvi_mipi.h>
#include <cvi_sns_ctrl.h>
// Sensor object (OV5647) to retrieve MIPI RX attributes directly
#include "../components/sophgo/video/include/app_ipcam_sensors.h"
}

// Global variables for frame capture control
std::atomic<bool> g_running(true);
std::atomic<int> g_frame_count(0);
std::chrono::steady_clock::time_point g_start_time;
std::chrono::steady_clock::time_point last_frame_time;

// Signal handler for graceful shutdown
void signalHandler(int signal)
{
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running.store(false);
}

// Ensure comprehensive cleanup on program exit
static void onExitCleanup()
{
    std::cout << "Exit cleanup: performing final video system reset..." << std::endl;
    deinitVideo();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

// Enhanced cleanup function to force reset video pipeline state
static void forceVideoSystemReset()
{
    std::cout << "Performing comprehensive video system reset..." << std::endl;
    
    // Try multiple cleanup attempts with delays
    for (int attempt = 0; attempt < 3; attempt++)
    {
        std::cout << "Cleanup attempt " << (attempt + 1) << "/3" << std::endl;
        deinitVideo();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Force try to destroy potential lingering VPSS groups
        try {
            for (int grp = 0; grp < 4; grp++) {  // Try common group numbers
                // Try to stop and destroy directly, ignore errors
                CVI_VPSS_StopGrp(grp);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                CVI_VPSS_DestroyGrp(grp);
            }
        } catch (...) {
            // Ignore any exceptions from forced cleanup
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    
    // Final longer delay for hardware to fully settle
    std::cout << "Waiting for hardware to settle..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

// Wrapper helpers
static inline bool startWrapperPipeline()
{
    // Enhanced cleanup: force reset multiple times with delays
    forceVideoSystemReset();
    
    if (initVideo() != 0)
    {
        std::cerr << "initVideo failed" << std::endl;
        return false;
    }
    // // Mirror example defaults: CH0 = H264 1920x1080@15, CH1 = JPEG 640x640@15 (preview)
    // video_ch_param_t ch0{};
    // ch0.format = VIDEO_FORMAT_H264;
    // ch0.width = 1920;
    // ch0.height = 1080;
    // ch0.fps = 15;
    // if (setupVideo(VIDEO_CH0, &ch0) != 0)
    // {
    //     std::cerr << "setupVideo CH0 failed" << std::endl;
    //     return false;
    // }

    video_ch_param_t ch1{};
    ch1.format = VIDEO_FORMAT_JPEG;
    ch1.width = 640;
    ch1.height = 480;
    ch1.fps = 15;
    if (setupVideo(VIDEO_CH1, &ch1) != 0)
    {
        std::cerr << "setupVideo CH1 failed" << std::endl;
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Try starting video with retry mechanism
    int video_start_attempts = 3;
    for (int attempt = 1; attempt <= video_start_attempts; attempt++)
    {
        std::cout << "Starting video system (attempt " << attempt << "/" << video_start_attempts << ")..." << std::endl;
        
        if (startVideo() == 0)
        {
            std::cout << "Video system started successfully!" << std::endl;
            return true;
        }
        
        std::cerr << "startVideo attempt " << attempt << " failed." << std::endl;
        
        if (attempt < video_start_attempts)
        {
            std::cerr << "Performing additional cleanup before retry..." << std::endl;
            deinitVideo();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
    
    std::cerr << "All startVideo attempts failed. Continuing anyway to attempt frame capture..." << std::endl;
    return true;
}

// TPU state
static ma::engine::EngineCVI *g_engine = nullptr;
static ma::Model *g_model = nullptr;

// CLI configuration (visible to frame callback)
static std::string g_cli_model_path;
static int g_cli_tpu_delay_ms = 0;
static bool g_cli_emit_base64 = true;
static bool g_cli_single_mode = false;
static std::chrono::steady_clock::time_point g_last_tpu_time = std::chrono::steady_clock::time_point::min();

static cv::Mat preprocessImage(cv::Mat &image, ma::Model *model)
{
    int ih = image.rows;
    int iw = image.cols;
    int oh = 0;
    int ow = 0;

    if (model->getInputType() == MA_INPUT_TYPE_IMAGE)
    {
        oh = reinterpret_cast<const ma_img_t *>(model->getInput())->height;
        ow = reinterpret_cast<const ma_img_t *>(model->getInput())->width;
    }

    cv::Mat resizedImage;
    double resize_scale = std::min((double)oh / ih, (double)ow / iw);
    int nh = (int)(ih * resize_scale);
    int nw = (int)(iw * resize_scale);
    cv::resize(image, resizedImage, cv::Size(nw, nh));
    int top = (oh - nh) / 2;
    int bottom = (oh - nh) - top;
    int left = (ow - nw) / 2;
    int right = (ow - nw) - left;

    cv::Mat paddedImage;
    cv::copyMakeBorder(resizedImage, paddedImage, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::cvtColor(paddedImage, paddedImage, cv::COLOR_BGR2RGB);

    return paddedImage;
}

int main(int argc, char** argv)
{
    std::string modelPath = "";
    int tpuDelay = 0;
    bool base64 = true;
    bool singleMode = false;

    CLI::App app{"CVITEK ReCamera Frame Capture - Wrapper Pipeline"};
    app.add_option("-m, --model", modelPath, "Path to the model file")->required();
    app.add_option("-t, --tpu-delay", tpuDelay, "How fast to process frames in the TPU (in milliseconds)")->default_str("0");
    app.add_option("-b, --base64", base64, "Output frames as base64. If false, base64 is omitted.")->default_str("true");
    app.add_option("-s, --single", singleMode, "Exit after one successful frame with TPU data")->default_str("false");
    CLI11_PARSE(app, argc, argv);

    // Copy to globals for callback use
    g_cli_model_path = modelPath;
    g_cli_tpu_delay_ms = tpuDelay;
    g_cli_emit_base64 = base64;
    g_cli_single_mode = singleMode;

    std::cout << "CVITEK ReCamera Frame Capture - Wrapper Pipeline" << std::endl;
    std::cout << "Will capture frames for 1 second and output as base64" << std::endl;

    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGKILL, signalHandler);
    signal(SIGQUIT, signalHandler);
    signal(SIGABRT, signalHandler);

    // Comprehensive initial cleanup - force reset any lingering state
    std::cout << "Performing initial system cleanup..." << std::endl;
    forceVideoSystemReset();
    std::atexit(onExitCleanup);

    // Start wrapper pipeline (working reference from example)
    if (!startWrapperPipeline())
    {
        std::cerr << "ERROR: Failed to start wrapper pipeline" << std::endl;
        deinitVideo();
        return -1;
    }
    std::cout << std::endl
              << "=== WRAPPER PIPELINE READY ===" << std::endl;
    std::cout << "Channels: CH1 JPEG 640x480@15" << std::endl;
    std::cout << "Starting capture..." << std::endl;
    std::cout << std::endl;

    // Validate model path exists
    {
        std::ifstream f(g_cli_model_path);
        if (!f.good()) {
            std::cerr << "Model file not found: " << g_cli_model_path << std::endl;
            return -1;
        }
    }

    // Init TPU
    g_engine = new ma::engine::EngineCVI();
    if (g_engine->init() != MA_OK)
    {
        std::cerr << "engine init failed" << std::endl;
    }
    else
    {
        if (g_cli_model_path.empty())
        {
            std::cerr << "no .cvimodel found (set MODEL_PATH or put a model under /userdata/MODEL/)" << std::endl;
        }
        else
        {
            if (g_engine->load(g_cli_model_path.c_str()) != MA_OK)
            {
                std::cerr << "engine load model failed: " << g_cli_model_path << std::endl;
            }
            else
            {
                g_model = ma::ModelFactory::create(g_engine);
                if (g_model == nullptr)
                {
                    std::cerr << "model not supported" << std::endl;
                }
            }
        }
    }

    // Re-register CH1 handler to append TPU output to JSON
    registerVideoFrameHandler(VIDEO_CH1, 0, [](void *pData, void *pArgs, void *pUserData) -> int
                              {
        static int warm = 0;
        VENC_STREAM_S* s = (VENC_STREAM_S*)pData;
        if (!s || s->u32PackCount == 0) return 0;
        if (warm < 30) { warm++; return 0; }
        
        // Calculate total frame size across all packs
        uint32_t total_len = 0;
        for (uint32_t i = 0; i < s->u32PackCount; i++) {
            VENC_PACK_S* pack = &s->pstPack[i];
            if (!pack->pu8Addr || pack->u32Len <= pack->u32Offset) {
                std::cerr << "Invalid pack " << i << " in stream" << std::endl;
                return 0;
            }
            total_len += (pack->u32Len - pack->u32Offset);
        }
        
        if (total_len == 0) {
            std::cerr << "No valid frame data found" << std::endl;
            return 0;
        }
        
        // Allocate buffer for complete frame
        std::vector<uint8_t> frame_data;
        frame_data.reserve(total_len);
        
        // Concatenate all packs into single frame
        for (uint32_t i = 0; i < s->u32PackCount; i++) {
            VENC_PACK_S* pack = &s->pstPack[i];
            uint8_t* pack_data = pack->pu8Addr + pack->u32Offset;
            uint32_t pack_len = pack->u32Len - pack->u32Offset;
            frame_data.insert(frame_data.end(), pack_data, pack_data + pack_len);
        }
        
        uint8_t* jpeg = frame_data.data();
        uint32_t len = frame_data.size();
        
        // Validate JPEG header (should start with 0xFF 0xD8)
        if (len < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8) {
            std::cerr << "Invalid JPEG header. First bytes: " << std::hex 
                     << (int)jpeg[0] << " " << (int)jpeg[1] << std::dec << std::endl;
            return 0;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time).count();
        auto elapsed_since_last_frame = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame_time).count();
        int idx = g_frame_count.fetch_add(1) + 1;
        std::string b64;
        if (g_cli_emit_base64) {
            try {
                b64 = macaron::Base64::Encode(std::string(reinterpret_cast<char*>(jpeg), len));
                if (b64.empty()) {
                    std::cerr << "Base64 encoding failed for frame " << idx << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Base64 encoding exception for frame " << idx << ": " << e.what() << std::endl;
                b64 = "";
            }
        }

        nlohmann::json j;
        j["frame_num"] = idx;
        j["elapsed"] = elapsed;
        j["elapsed_since_last_frame"] = elapsed_since_last_frame;
        j["jpeg_size"] = len;
        j["pack_count"] = s->u32PackCount;
        j["total_packs_size"] = total_len;
        if (g_cli_emit_base64 && !b64.empty()) {
            j["frame"] = b64;
        } else if (g_cli_emit_base64) {
            j["frame_error"] = "base64_encoding_failed";
        }

        bool do_tpu = true;
        if (g_cli_tpu_delay_ms > 0 && g_last_tpu_time != std::chrono::steady_clock::time_point::min()) {
            auto since_last = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_last_tpu_time).count();
            if (since_last < g_cli_tpu_delay_ms) {
                do_tpu = false;
            }
        }

        if (do_tpu && g_engine && g_model && g_model->getInputType() == MA_INPUT_TYPE_IMAGE) {
            try {
                std::vector<uint8_t> buf(jpeg, jpeg + len);
                cv::Mat im = cv::imdecode(buf, cv::IMREAD_COLOR);
                if (!im.empty()) {
                    cv::Mat pre = preprocessImage(im, g_model);

                    ma_img_t img;
                    img.data   = (uint8_t*)pre.data;
                    img.size   = pre.rows * pre.cols * pre.channels();
                    img.width  = pre.cols;
                    img.height = pre.rows;
                    img.format = MA_PIXEL_FORMAT_RGB888;
                    img.rotate = MA_PIXEL_ROTATE_0;

                    ma_err_t ret = MA_OK;
                    ma::model::Classifier* classifier = nullptr;
                    ma::model::PoseDetector* pose = nullptr;
                    ma::model::Segmentor* seg = nullptr;
                    ma::model::Detector* det = nullptr;
                    if (g_model->getOutputType() == MA_OUTPUT_TYPE_CLASS) {
                        classifier = static_cast<ma::model::Classifier*>(g_model);
                        ret = classifier->run(&img);
                    } else if (g_model->getOutputType() == MA_OUTPUT_TYPE_KEYPOINT) {
                        pose = static_cast<ma::model::PoseDetector*>(g_model);
                        ret = pose->run(&img);
                    } else if (g_model->getOutputType() == MA_OUTPUT_TYPE_SEGMENT) {
                        seg = static_cast<ma::model::Segmentor*>(g_model);
                        ret = seg->run(&img);
                    } else if (g_model->getOutputType() == MA_OUTPUT_TYPE_BBOX || g_model->getOutputType() == MA_OUTPUT_TYPE_TENSOR) {
                        det = static_cast<ma::model::Detector*>(g_model);
                        ret = det->run(&img);
                    } else {
                        ret = MA_ENOTSUP;
                    }

                    if (ret == MA_OK) {
                        auto tpu_now = std::chrono::steady_clock::now();
                        auto tpu_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tpu_now - now).count();
                        nlohmann::json tout = nlohmann::json::array();
                        int outs = g_engine->getOutputSize();
                        for (int i = 0; i < outs; ++i) {
                            auto t = g_engine->getOutput(i);
                            nlohmann::json tj;
                            tj["index"] = i;
                            tj["size"] = (uint64_t)t.size;
                            tj["type"] = (int)t.type;
                            nlohmann::json shape = nlohmann::json::array();
                            for (uint32_t d = 0; d < t.shape.size; ++d) shape.push_back(t.shape.dims[d]);
                            tj["shape"] = shape;
                            tj["quant_scale"] = t.quant_param.scale;
                            tj["quant_zero"] = t.quant_param.zero_point;
                            tj["is_physical"] = t.is_physical;
                            tj["is_variable"] = t.is_variable;
                            tj["name"] = t.name;
                            // std::string raw(reinterpret_cast<const char*>(t.data.data), t.size);
                            // tj["data_b64"] = macaron::Base64::Encode(raw);
                            tout.push_back(tj);
                        }
                        j["tpu"] = tout;
                        j["tpu_elapsed"] = tpu_elapsed;
                        g_last_tpu_time = now;
                        j["output_type"] = g_model->getOutputType();
                        j["model_name"] = g_model->getName();
                        j["model_type"] = g_model->getType();
                        j["model_input_type"] = g_model->getInputType();

                        // Parsed results (like example), if available
                        nlohmann::json results = nlohmann::json::array();
                        if (classifier) {
                            const auto& r = classifier->getResults();
                            for (const auto& it : r) {
                                nlohmann::json cj;
                                cj["kind"] = "classifier";
                                cj["score"] = it.score;
                                cj["target"] = it.target;
                                results.push_back(cj);
                            }
                        } else if (det) {
                            const auto& r = det->getResults();
                            for (const auto& it : r) {
                                nlohmann::json dj;
                                dj["kind"] = "detector";
                                dj["x"] = it.x;
                                dj["y"] = it.y;
                                dj["w"] = it.w;
                                dj["h"] = it.h;
                                dj["score"] = it.score;
                                dj["target"] = it.target;
                                // absolute coords in preprocessed image space
                                float x1 = (it.x - it.w / 2.0f) * pre.cols;
                                float y1 = (it.y - it.h / 2.0f) * pre.rows;
                                float x2 = (it.x + it.w / 2.0f) * pre.cols;
                                float y2 = (it.y + it.h / 2.0f) * pre.rows;
                                dj["abs"] = { {"x1", x1}, {"y1", y1}, {"x2", x2}, {"y2", y2} };
                                results.push_back(dj);
                            }
                        } else if (pose) {
                            const auto& r = pose->getResults();
                            for (const auto& it : r) {
                                nlohmann::json pj;
                                pj["kind"] = "pose";
                                pj["box"] = {
                                    {"x", it.box.x}, {"y", it.box.y}, {"w", it.box.w}, {"h", it.box.h},
                                    {"score", it.box.score}, {"target", it.box.target}
                                };
                                float x1 = (it.box.x - it.box.w / 2.0f) * pre.cols;
                                float y1 = (it.box.y - it.box.h / 2.0f) * pre.rows;
                                float x2 = (it.box.x + it.box.w / 2.0f) * pre.cols;
                                float y2 = (it.box.y + it.box.h / 2.0f) * pre.rows;
                                pj["box_abs"] = { {"x1", x1}, {"y1", y1}, {"x2", x2}, {"y2", y2} };
                                nlohmann::json pts = nlohmann::json::array();
                                for (const auto& pt : it.pts) {
                                    pts.push_back({ {"x", pt.x}, {"y", pt.y}, {"z", pt.z} });
                                }
                                pj["keypoints"] = pts;
                                results.push_back(pj);
                            }
                        } else if (seg) {
                            const auto& r = seg->getResults();
                            for (const auto& it : r) {
                                nlohmann::json sj;
                                sj["kind"] = "segment";
                                sj["box"] = {
                                    {"x", it.box.x}, {"y", it.box.y}, {"w", it.box.w}, {"h", it.box.h},
                                    {"score", it.box.score}, {"target", it.box.target}
                                };
                                float x1 = (it.box.x - it.box.w / 2.0f) * pre.cols;
                                float y1 = (it.box.y - it.box.h / 2.0f) * pre.rows;
                                float x2 = (it.box.x + it.box.w / 2.0f) * pre.cols;
                                float y2 = (it.box.y + it.box.h / 2.0f) * pre.rows;
                                sj["box_abs"] = { {"x1", x1}, {"y1", y1}, {"x2", x2}, {"y2", y2} };
                                sj["mask"] = { {"width", it.mask.width}, {"height", it.mask.height}, {"data", it.mask.data} };
                                results.push_back(sj);
                            }
                        }
                        j["results"] = results;
                        j["pre_shape"] = { {"w", pre.cols}, {"h", pre.rows} };
                        
                        // Exit after one successful frame in single mode
                        if (g_cli_single_mode) {
                            g_running.store(false);
                        }
                    } else {
                        j["tpu_err"] = ret;
                    }
                }
            } catch (...) {
                j["tpu_err"] = -1;
            }
        }

        std::cout << j.dump() << std::endl;
        last_frame_time = std::chrono::steady_clock::now();
        return 0;
    }, nullptr);

    g_start_time = std::chrono::steady_clock::now();
    while (g_running.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << std::endl
              << "=== CAPTURE COMPLETED ===" << std::endl;

    std::cout << "Performing comprehensive cleanup..." << std::endl;
    
    // Clean up TPU resources first
    if (g_model)
    {
        ma::ModelFactory::remove(g_model);
        g_model = nullptr;
    }
    if (g_engine) 
    {
        delete g_engine;
        g_engine = nullptr;
    }
    
    // Comprehensive video system cleanup
    std::cout << "Cleaning up video system..." << std::endl;
    deinitVideo();
    
    // Additional delay and force cleanup to ensure clean state for next run
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << "Program finished successfully." << std::endl;
    return 0;
}