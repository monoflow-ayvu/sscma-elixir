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

// Signal handler for graceful shutdown
void signalHandler(int signal)
{
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running.store(false);
}

// Ensure video deinitialization on normal program exit
static void onExitCleanup()
{
    deinitVideo();
}

// Wrapper helpers
static inline bool startWrapperPipeline()
{
    // Mirror example: ensure clean state, then init, setup, register, start with small delays
    deinitVideo();
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
    if (startVideo() != 0)
    {
        std::cerr << "startVideo failed. Trying to continue anyway." << std::endl;
    }
    return true;
}

// TPU state
static ma::engine::EngineCVI *g_engine = nullptr;
static ma::Model *g_model = nullptr;

static std::string findModelPath()
{
    const char *envPath = std::getenv("MODEL_PATH");
    if (envPath && *envPath)
        return std::string(envPath);

    const char *kModelDir = "/userdata/MODEL/";
    std::vector<cv::String> files;
    cv::glob(std::string(kModelDir) + "/*.cvimodel", files, false);
    if (!files.empty())
        return std::string(files[0]);
    return std::string();
}

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

int main()
{
    std::cout << "CVITEK ReCamera Frame Capture - Wrapper Pipeline" << std::endl;
    std::cout << "Will capture frames for 1 second and output as base64" << std::endl;

    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGKILL, signalHandler);
    signal(SIGQUIT, signalHandler);
    signal(SIGABRT, signalHandler);

    // Ensure wrapper state is clean
    deinitVideo();
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
    std::cout << "Channels: CH0 H264 1920x1080@15, CH1 JPEG 640x480@15" << std::endl;
    std::cout << "Starting capture..." << std::endl;
    std::cout << std::endl;

    // Init TPU
    g_engine = new ma::engine::EngineCVI();
    if (g_engine->init() != MA_OK)
    {
        std::cerr << "engine init failed" << std::endl;
    }
    else
    {
        std::string modelPath = findModelPath();
        if (modelPath.empty())
        {
            std::cerr << "no .cvimodel found (set MODEL_PATH or put a model under /userdata/MODEL/)" << std::endl;
        }
        else
        {
            if (g_engine->load(modelPath.c_str()) != MA_OK)
            {
                std::cerr << "engine load model failed: " << modelPath << std::endl;
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
        VENC_PACK_S* pack = &s->pstPack[0];
        uint8_t* jpeg = pack->pu8Addr + pack->u32Offset;
        uint32_t len = pack->u32Len - pack->u32Offset;
        if (!jpeg || len == 0) return 0;
        if (warm < 30) { warm++; return 0; }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time).count();
        int idx = g_frame_count.fetch_add(1) + 1;
        std::string b64 = macaron::Base64::Encode(std::string(reinterpret_cast<char*>(jpeg), len));

        nlohmann::json j;
        j["frame_num"] = idx;
        j["elapsed"] = elapsed;
        j["jpeg_size"] = len;
        j["base64_str"] = b64;

        if (g_engine && g_model && g_model->getInputType() == MA_INPUT_TYPE_IMAGE) {
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
                    if (g_model->getOutputType() == MA_OUTPUT_TYPE_CLASS) {
                        auto* classifier = static_cast<ma::model::Classifier*>(g_model);
                        ret = classifier->run(&img);
                    } else if (g_model->getOutputType() == MA_OUTPUT_TYPE_KEYPOINT) {
                        auto* pose = static_cast<ma::model::PoseDetector*>(g_model);
                        ret = pose->run(&img);
                    } else if (g_model->getOutputType() == MA_OUTPUT_TYPE_SEGMENT) {
                        auto* seg = static_cast<ma::model::Segmentor*>(g_model);
                        ret = seg->run(&img);
                    } else if (g_model->getOutputType() == MA_OUTPUT_TYPE_BBOX || g_model->getOutputType() == MA_OUTPUT_TYPE_TENSOR) {
                        auto* det = static_cast<ma::model::Detector*>(g_model);
                        ret = det->run(&img);
                    } else {
                        ret = MA_ENOTSUP;
                    }

                    if (ret == MA_OK) {
                        auto tpu_now = std::chrono::steady_clock::now();
                        auto tpu_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tpu_now - g_start_time).count();
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
                            // std::string raw(reinterpret_cast<const char*>(t.data.data), t.size);
                            // tj["data_b64"] = macaron::Base64::Encode(raw);
                            tout.push_back(tj);
                        }
                        j["tpu"] = tout;
                        j["tpu_elapsed"] = tpu_elapsed;
                    } else {
                        j["tpu_err"] = ret;
                    }

                    g_running.store(false);
                }
            } catch (...) {
                j["tpu_err"] = -1;
            }
        }

        std::cout << j.dump() << std::endl;
        return 0; }, nullptr);

    g_start_time = std::chrono::steady_clock::now();
    while (g_running.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << std::endl
              << "=== CAPTURE COMPLETED ===" << std::endl;

    std::cout << "Cleaning up video system..." << std::endl;
    deinitVideo();

    if (g_model)
    {
        ma::ModelFactory::remove(g_model);
        g_model = nullptr;
    }
    delete g_engine;
    g_engine = nullptr;

    std::cout << "Program finished successfully." << std::endl;
    return 0;
}