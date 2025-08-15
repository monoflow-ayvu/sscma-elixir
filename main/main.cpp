#include <chrono>
#include <iostream>
#include <signal.h>
#include <string>
#include <atomic>
#include <unistd.h>
#include <string.h>
#include <thread>

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
static std::thread g_isp_thread;
static bool g_isp_thread_started = false;
static VB_POOL g_vi_vb_pool = VB_INVALID_POOLID;

#ifndef ALIGN_UP
#define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((a) - 1))
#endif

// Camera and system configuration
#define VI_DEV_ID 0
#define VI_PIPE_ID 0
#define VI_CHN_ID 0
#define VPSS_GRP_ID 0
#define VPSS_CHN_ID 0
#define VENC_CHN_ID 0

// Image parameters
#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define FRAME_RATE 30

// Signal handler for graceful shutdown
void signalHandler(int signal)
{
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running.store(false);
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
    // Mirror example defaults: CH0 = H264 1920x1080@15, CH1 = JPEG 640x640@15 (preview)
    video_ch_param_t ch0{};
    ch0.format = VIDEO_FORMAT_H264;
    ch0.width = 1920;
    ch0.height = 1080;
    ch0.fps = 15;
    if (setupVideo(VIDEO_CH0, &ch0) != 0)
    {
        std::cerr << "setupVideo CH0 failed" << std::endl;
        return false;
    }

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

    if (registerVideoFrameHandler(VIDEO_CH1, 0, [](void *pData, void *pArgs, void *pUserData) -> int
                                  {
        static int warm = 0;
        VENC_STREAM_S* s = (VENC_STREAM_S*)pData;
        if (!s || s->u32PackCount == 0) return 0;
        VENC_PACK_S* pack = &s->pstPack[0];
        uint8_t* jpeg = pack->pu8Addr + pack->u32Offset;
        uint32_t len = pack->u32Len - pack->u32Offset;
        if (!jpeg || len == 0) return 0;
        // Warm-up frames (auto-exposure/awb settle)
        if (warm < 30) { warm++; return 0; }
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time).count();
        int idx = g_frame_count.fetch_add(1) + 1;
        std::string b64 = macaron::Base64::Encode(std::string(reinterpret_cast<char*>(jpeg), len));
        // std::cout << std::endl << std::endl << "Frame " << idx << " [" << elapsed << "ms] Size: " << len << " bytes\n";
        // std::cout << "data:image/jpeg;base64," << b64 << "\n\n";
        nlohmann::json j;
        j["frame_num"] = idx;
        j["elapsed"] = elapsed;
        j["jpeg_size"] = len;
        j["base64_str"] = b64;
        std::cout << j.dump() << std::endl;
        return 0; }, nullptr) != 0)
    {
        std::cerr << "registerVideoFrameHandler failed" << std::endl;
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (startVideo() != 0)
    {
        std::cerr << "startVideo failed. Trying to continue anyway." << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return true;
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

    g_start_time = std::chrono::steady_clock::now();
    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << std::endl
              << "=== CAPTURE COMPLETED ===" << std::endl;

    std::cout << "Cleaning up video system..." << std::endl;
    deinitVideo();

    std::cout << "Program finished successfully." << std::endl;
    return 0;
}