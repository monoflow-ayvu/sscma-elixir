#include <chrono>
#include <iostream>
#include <signal.h>
#include <string>
#include <atomic>
#include <unistd.h>
#include <string.h>
#include <thread>

#include "base64.hpp"
extern "C" {
#include <video.h>
}

extern "C" {
// Core CVITEK headers - direct approach only
#include <cvi_vb.h>
#include <cvi_sys.h>
#include <cvi_vi.h>
#include <cvi_comm_vi.h>
#include <cvi_vpss.h>
#include <cvi_venc.h>
#include <cvi_comm_venc.h>
// ISP / AE / AWB / MIPI (direct APIs)
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
#define ALIGN_UP(x, a)   (((x) + ((a) - 1)) & ~((a) - 1))
#endif

// Camera and system configuration
#define VI_DEV_ID       0
#define VI_PIPE_ID      0  
#define VI_CHN_ID       0
#define VPSS_GRP_ID     0
#define VPSS_CHN_ID     0
#define VENC_CHN_ID     0

// Image parameters
#define IMAGE_WIDTH     1920
#define IMAGE_HEIGHT    1080
#define FRAME_RATE      30

// Signal handler for graceful shutdown
void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running.store(false);
}

// Force cleanup of any existing state - critical for multiple runs
void checkAndCleanupSystems() {
    std::cout << "Cautiously cleaning up any existing system state..." << std::endl;
    
    // Cautiously cleanup VENC (ignore errors if not initialized)
    CVI_VENC_StopRecvFrame(VENC_CHN_ID);
    CVI_VENC_DestroyChn(VENC_CHN_ID);
    
    // Cautiously cleanup VPSS (ignore errors if not initialized)
    CVI_VPSS_DisableChn(VPSS_GRP_ID, VPSS_CHN_ID);
    CVI_VPSS_StopGrp(VPSS_GRP_ID);
    CVI_VPSS_DestroyGrp(VPSS_GRP_ID);
    
    // Cautiously cleanup VI (ignore errors if not initialized)
    CVI_VI_DisableChn(VI_PIPE_ID, VI_CHN_ID);
    CVI_VI_StopPipe(VI_PIPE_ID);
    CVI_VI_DestroyPipe(VI_PIPE_ID);
    CVI_VI_DisableDev(VI_DEV_ID);
    
    // Unbind any remaining pipeline connections (ignore errors)
    MMF_CHN_S stSrcChn, stDestChn;
    
    // Try to unbind VPSS -> VENC
    stSrcChn = {CVI_ID_VPSS, VPSS_GRP_ID, VPSS_CHN_ID};
    stDestChn = {CVI_ID_VENC, 0, VENC_CHN_ID};
    CVI_SYS_UnBind(&stSrcChn, &stDestChn);
    
    // Try to unbind VI -> VPSS
    stSrcChn = {CVI_ID_VI, VI_PIPE_ID, VI_CHN_ID};
    stDestChn = {CVI_ID_VPSS, VPSS_GRP_ID, 0};
    CVI_SYS_UnBind(&stSrcChn, &stDestChn);
    
    // Close VI system (ignore errors)
    CVI_SYS_VI_Close();
    
    // Try to exit SYS (this will fail gracefully if not initialized)
    CVI_SYS_Exit();
    
    // Try to exit VB (this will fail gracefully if not initialized) 
    CVI_VB_Exit();
    
    std::cout << "Cautious cleanup completed (errors ignored)" << std::endl;
}

// Initialize VB (Video Buffer) system with proper configuration
CVI_S32 initVbSystem() {
    std::cout << "Initializing VB system..." << std::endl;
    
    VB_CONFIG_S stVbConf;
    memset(&stVbConf, 0, sizeof(VB_CONFIG_S));
    
    // Configure 4 VB pools sized to realistic requirements
    stVbConf.u32MaxPoolCnt = 4;
    
    // Pool 0: VI raw sensor data (Bayer ~2 bytes per pixel for 10/12bpp)
    stVbConf.astCommPool[0].u32BlkSize = IMAGE_WIDTH * IMAGE_HEIGHT * 2;
    stVbConf.astCommPool[0].u32BlkCnt = 8;
    stVbConf.astCommPool[0].enRemapMode = VB_REMAP_MODE_CACHED;
    
    // Pool 1: ISP processed YUV420 (w*h*3/2)
    stVbConf.astCommPool[1].u32BlkSize = IMAGE_WIDTH * IMAGE_HEIGHT * 3 / 2;
    stVbConf.astCommPool[1].u32BlkCnt = 8;
    stVbConf.astCommPool[1].enRemapMode = VB_REMAP_MODE_CACHED;
    
    // Pool 2: VPSS processed YUV420
    stVbConf.astCommPool[2].u32BlkSize = IMAGE_WIDTH * IMAGE_HEIGHT * 3 / 2;
    stVbConf.astCommPool[2].u32BlkCnt = 6;
    stVbConf.astCommPool[2].enRemapMode = VB_REMAP_MODE_CACHED;
    
    // Pool 3: VENC output (JPEG) - align to 1024 (allocate ~2MB for 1080p)
    CVI_U32 u32JpegBufSize = 2 * 1024 * 1024;
    stVbConf.astCommPool[3].u32BlkSize = (u32JpegBufSize + 1023) & ~1023; // Align to 1024
    stVbConf.astCommPool[3].u32BlkCnt = 4;
    stVbConf.astCommPool[3].enRemapMode = VB_REMAP_MODE_CACHED;
    
    CVI_S32 ret = CVI_VB_SetConfig(&stVbConf);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VB_SetConfig failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    ret = CVI_VB_Init();
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VB_Init failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    std::cout << "VB system initialized with 4 pools (larger buffers for ISP)" << std::endl;
  return CVI_SUCCESS;
}

// Initialize SYS (System) module
CVI_S32 initSysSystem() {
    std::cout << "Initializing SYS system..." << std::endl;
    
    CVI_S32 ret = CVI_SYS_Init();
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_SYS_Init failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    std::cout << "SYS system initialized successfully" << std::endl;
    return CVI_SUCCESS;
}

// Initialize VI (Video Input) with minimal configuration for direct API approach
CVI_S32 initVideoInput() {
    std::cout << "Initializing VI system with minimal configuration..." << std::endl;
    
    // Step 1: Open VI system
    CVI_S32 ret = CVI_SYS_VI_Open();
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_SYS_VI_Open failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    // Step 2: Configure MIPI RX for OV5647 (reset, set attr, enable clock)
    {
        std::cout << "MIPI: Configure RX for OV5647..." << std::endl;
        ISP_SNS_OBJ_S *pSnsObj = &stSnsOv5647_Obj;
        SNS_COMBO_DEV_ATTR_S combo_attr;
        memset(&combo_attr, 0, sizeof(combo_attr));
        if (pSnsObj) {
            // Set sensor bus info (I2C bus 2 for OV5647 on reCamera per SDK params)
            ISP_SNS_COMMBUS_U busInfo; memset(&busInfo, 0, sizeof(busInfo));
            busInfo.s8I2cDev = 2; // matches components/sophgo/video params
            if (pSnsObj->pfnSetBusInfo) {
                CVI_S32 bret = pSnsObj->pfnSetBusInfo(VI_PIPE_ID, busInfo);
                if (bret != CVI_SUCCESS) {
                    std::cerr << "pfnSetBusInfo failed: 0x" << std::hex << bret << std::endl;
                }
            }

            CVI_S32 mret = pSnsObj->pfnGetRxAttr ? pSnsObj->pfnGetRxAttr(VI_PIPE_ID, &combo_attr) : CVI_FAILURE;
            if (mret != CVI_SUCCESS) {
                std::cerr << "pfnGetRxAttr failed: 0x" << std::hex << mret << " (continuing without explicit MIPI attr)" << std::endl;
            } else {
                MIPI_DEV mipiDev = combo_attr.devno;
                CVI_MIPI_SetSensorReset(mipiDev, 1);
                CVI_MIPI_SetMipiReset(mipiDev, 1);
                mret = CVI_MIPI_SetMipiAttr(mipiDev, (void*)&combo_attr);
                if (mret != CVI_SUCCESS) {
                    std::cerr << "CVI_MIPI_SetMipiAttr failed: 0x" << std::hex << mret << " (continuing)" << std::endl;
                }
                CVI_MIPI_SetSensorClock(mipiDev, 1);
                usleep(20);
                CVI_MIPI_SetSensorReset(mipiDev, 0);
                if (pSnsObj->pfnSnsProbe) {
                    (void)pSnsObj->pfnSnsProbe(mipiDev);
                }
                std::cout << "MIPI: RX configured (dev=" << (int)mipiDev << ")" << std::endl;
            }
        } else {
            std::cerr << "OV5647 sensor object or pfnGetRxAttr not available" << std::endl;
        }
    }

    // Step 3: Configure VI device for OV5647 sensor with proper AD channel setup
    VI_DEV_ATTR_S stViDevAttr;
    memset(&stViDevAttr, 0, sizeof(VI_DEV_ATTR_S));
    
    // Critical: Set exact frame size that matches sensor output  
    stViDevAttr.stSize.u32Width = IMAGE_WIDTH;   // 1920
    stViDevAttr.stSize.u32Height = IMAGE_HEIGHT; // 1080
    stViDevAttr.enIntfMode = VI_MODE_MIPI;
    stViDevAttr.enWorkMode = VI_WORK_MODE_1Multiplex;
    stViDevAttr.enScanMode = VI_SCAN_PROGRESSIVE;
    
    // CRITICAL FIX: Use proper AD channel (0, not -1) for OV5647
    stViDevAttr.as32AdChnId[0] = 0;  // First AD channel active
    stViDevAttr.as32AdChnId[1] = -1; // Others disabled
    stViDevAttr.as32AdChnId[2] = -1;
    stViDevAttr.as32AdChnId[3] = -1;
    
    // Data configuration
    stViDevAttr.enDataSeq = VI_DATA_SEQ_YUYV;
    stViDevAttr.enInputDataType = VI_DATA_TYPE_RGB;
    stViDevAttr.enBayerFormat = BAYER_FORMAT_BG;
    stViDevAttr.stWDRAttr.enWDRMode = WDR_MODE_NONE;
    stViDevAttr.stWDRAttr.u32CacheLine = IMAGE_HEIGHT;
    
    ret = CVI_VI_SetDevAttr(VI_DEV_ID, &stViDevAttr);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VI_SetDevAttr failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    ret = CVI_VI_EnableDev(VI_DEV_ID);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VI_EnableDev failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    // Step 2.5: Device timing will be auto-detected by sensor configuration
    
    // Step 4: Create VI pipe with minimal but proper configuration
    VI_PIPE_ATTR_S stPipeAttr;
    memset(&stPipeAttr, 0, sizeof(VI_PIPE_ATTR_S));
    stPipeAttr.enPipeBypassMode = VI_PIPE_BYPASS_NONE;
    stPipeAttr.bYuvSkip = CVI_FALSE;
    stPipeAttr.bIspBypass = CVI_FALSE;  // Don't bypass ISP - causes frame width issues
    stPipeAttr.u32MaxW = IMAGE_WIDTH;
    stPipeAttr.u32MaxH = IMAGE_HEIGHT; 
    stPipeAttr.enPixFmt = PIXEL_FORMAT_RGB_BAYER_10BPP;
    stPipeAttr.enBitWidth = DATA_BITWIDTH_10;
    stPipeAttr.enCompressMode = COMPRESS_MODE_NONE;
    stPipeAttr.bNrEn = CVI_TRUE;  // Enable NR for proper processing
    stPipeAttr.bSharpenEn = CVI_FALSE;
    stPipeAttr.stFrameRate.s32SrcFrameRate = FRAME_RATE;
    stPipeAttr.stFrameRate.s32DstFrameRate = FRAME_RATE;
    stPipeAttr.bDiscardProPic = CVI_FALSE;
    stPipeAttr.bYuvBypassPath = CVI_FALSE;  // Disable YUV bypass for proper processing 
    
    ret = CVI_VI_CreatePipe(VI_PIPE_ID, &stPipeAttr);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VI_CreatePipe failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    // Step 5: CRITICAL - Bind device to pipe for proper sensor connection
    ret = CVI_VI_SetDevBindPipe(VI_DEV_ID, VI_PIPE_ID);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VI_SetDevBindPipe failed: 0x" << std::hex << ret << std::endl;
        // Continue anyway - some configurations may not require explicit binding
    }

    // Step 5: Use default pipe frame source (device)

    // Step 6: Initialize ISP and start processing pipeline before starting pipe
    {
        ISP_PUB_ATTR_S stPubAttr;
        memset(&stPubAttr, 0, sizeof(ISP_PUB_ATTR_S));
        stPubAttr.enBayer = (ISP_BAYER_FORMAT_E)BAYER_FORMAT_BG;
        stPubAttr.f32FrameRate = FRAME_RATE;
        stPubAttr.enWDRMode = WDR_MODE_NONE;
        stPubAttr.stWndRect.s32X = 0;
        stPubAttr.stWndRect.s32Y = 0;
        stPubAttr.stWndRect.u32Width = IMAGE_WIDTH;
        stPubAttr.stWndRect.u32Height = IMAGE_HEIGHT;
        stPubAttr.stSnsSize.u32Width = IMAGE_WIDTH;
        stPubAttr.stSnsSize.u32Height = IMAGE_HEIGHT;

        std::cout << "ISP: MemInit..." << std::endl;
        CVI_S32 iret = CVI_ISP_MemInit(VI_PIPE_ID);
        if (iret != CVI_SUCCESS) {
            std::cerr << "CVI_ISP_MemInit failed: 0x" << std::hex << iret << std::endl;
            return iret;
        }

        std::cout << "ISP: SetPubAttr..." << std::endl;
        iret = CVI_ISP_SetPubAttr(VI_PIPE_ID, &stPubAttr);
        if (iret != CVI_SUCCESS) {
            std::cerr << "CVI_ISP_SetPubAttr failed: 0x" << std::hex << iret << std::endl;
            return iret;
        }

        std::cout << "ISP: Init..." << std::endl;
        iret = CVI_ISP_Init(VI_PIPE_ID);
        if (iret != CVI_SUCCESS) {
            std::cerr << "CVI_ISP_Init failed: 0x" << std::hex << iret << std::endl;
            return iret;
        }

        // Register AE/AWB libraries and sensor callbacks
        ALG_LIB_S stAeLib; memset(&stAeLib, 0, sizeof(ALG_LIB_S));
        ALG_LIB_S stAwbLib; memset(&stAwbLib, 0, sizeof(ALG_LIB_S));
        strncpy(stAeLib.acLibName, "ae_lib", sizeof(stAeLib.acLibName)-1);
        strncpy(stAwbLib.acLibName, "awb_lib", sizeof(stAwbLib.acLibName)-1);
        stAeLib.s32Id = VI_PIPE_ID; stAwbLib.s32Id = VI_PIPE_ID;
        (void)CVI_AE_Register(VI_PIPE_ID, &stAeLib);
        (void)CVI_AWB_Register(VI_PIPE_ID, &stAwbLib);
        if (stSnsOv5647_Obj.pfnRegisterCallback) {
            (void)stSnsOv5647_Obj.pfnRegisterCallback(VI_PIPE_ID, &stAeLib, &stAwbLib);
        }

        std::cout << "ISP: Run (thread)..." << std::endl;
        g_isp_thread_started = true;
        g_isp_thread = std::thread([](){
            CVI_S32 r = CVI_ISP_Run(VI_PIPE_ID);
            if (r != CVI_SUCCESS) {
                std::cerr << "CVI_ISP_Run thread exit with error: 0x" << std::hex << r << std::endl;
            }
        });
    }
    
    // Step 8: Start pipe after ISP is running (log before/after)
    std::cout << "VI: StartPipe..." << std::endl;
    ret = CVI_VI_StartPipe(VI_PIPE_ID);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VI_StartPipe failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    std::cout << "VI: StartPipe OK" << std::endl;
    
    // Allow ISP/sensor to stabilize
    usleep(200000); // 200ms
    
    // Step 9: Configure VI channel for YUV output
    VI_CHN_ATTR_S stChnAttr;
    memset(&stChnAttr, 0, sizeof(VI_CHN_ATTR_S));
    stChnAttr.stSize.u32Width = IMAGE_WIDTH;
    stChnAttr.stSize.u32Height = IMAGE_HEIGHT;
    stChnAttr.enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
    stChnAttr.enDynamicRange = DYNAMIC_RANGE_SDR8;
    stChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    stChnAttr.enCompressMode = COMPRESS_MODE_NONE;
    stChnAttr.bMirror = CVI_FALSE;
    stChnAttr.bFlip = CVI_FALSE;
    stChnAttr.u32Depth = 2;  // Frame buffer depth
    stChnAttr.stFrameRate.s32SrcFrameRate = -1;
    stChnAttr.stFrameRate.s32DstFrameRate = -1;
    
    ret = CVI_VI_SetChnAttr(VI_PIPE_ID, VI_CHN_ID, &stChnAttr);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VI_SetChnAttr failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    // Create and attach a dedicated VB pool for VI channel to avoid large ION allocations
    {
        CVI_U32 stride = ALIGN_UP(IMAGE_WIDTH, 64);
        CVI_U32 y_size = stride * IMAGE_HEIGHT;
        CVI_U32 uv_size = y_size / 2;
        CVI_U32 blk_size = y_size + uv_size; // YUV420
        // Increase to 4MB to match internal VI DMA buffer alignment/margins
        if (blk_size < (4 * 1024 * 1024)) blk_size = 4 * 1024 * 1024;
        blk_size = ALIGN_UP(blk_size, 1024);

        VB_POOL_CONFIG_S stPoolCfg;
        memset(&stPoolCfg, 0, sizeof(stPoolCfg));
        stPoolCfg.u32BlkSize = blk_size;
        stPoolCfg.u32BlkCnt = 4;
        stPoolCfg.enRemapMode = VB_REMAP_MODE_CACHED;

        std::cout << "VI: Create VB pool blk_size=" << blk_size << " blk_cnt=4" << std::endl;
        g_vi_vb_pool = CVI_VB_CreatePool(&stPoolCfg);
        if (g_vi_vb_pool == VB_INVALID_POOLID) {
            std::cerr << "CVI_VB_CreatePool for VI failed" << std::endl;
        } else {
            CVI_S32 aRet = CVI_VI_AttachVbPool(VI_PIPE_ID, VI_CHN_ID, g_vi_vb_pool);
            if (aRet != CVI_SUCCESS) {
                std::cerr << "CVI_VI_AttachVbPool failed: 0x" << std::hex << aRet << std::endl;
            } else {
                std::cout << "VI: Attached VB pool successfully" << std::endl;
            }
        }
    }

    std::cout << "VI: EnableChn..." << std::endl;
    ret = CVI_VI_EnableChn(VI_PIPE_ID, VI_CHN_ID);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VI_EnableChn failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    std::cout << "VI: EnableChn OK" << std::endl;

    // Short delay before downstream bindings
    usleep(100000); // 100ms
    
    std::cout << "VI system initialized successfully (device bound, timing configured, ISP enabled)" << std::endl;
    return CVI_SUCCESS;
}

// Initialize VPSS (Video Processing SubSystem)
CVI_S32 initVideoProcessing() {
    std::cout << "Initializing VPSS system..." << std::endl;
    
    // Step 1: Set VPSS group attributes
    VPSS_GRP_ATTR_S stVpssGrpAttr;
    memset(&stVpssGrpAttr, 0, sizeof(VPSS_GRP_ATTR_S));
    stVpssGrpAttr.stFrameRate.s32SrcFrameRate = FRAME_RATE;
    stVpssGrpAttr.stFrameRate.s32DstFrameRate = FRAME_RATE;
    stVpssGrpAttr.enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
    stVpssGrpAttr.u32MaxW = IMAGE_WIDTH;
    stVpssGrpAttr.u32MaxH = IMAGE_HEIGHT;
    stVpssGrpAttr.u8VpssDev = 0;
    
    CVI_S32 ret = CVI_VPSS_CreateGrp(VPSS_GRP_ID, &stVpssGrpAttr);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VPSS_CreateGrp failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    // Step 2: Set VPSS channel attributes
    VPSS_CHN_ATTR_S stVpssChnAttr;
    memset(&stVpssChnAttr, 0, sizeof(VPSS_CHN_ATTR_S));
    stVpssChnAttr.u32Width = IMAGE_WIDTH;
    stVpssChnAttr.u32Height = IMAGE_HEIGHT;
    stVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    stVpssChnAttr.enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
    stVpssChnAttr.stFrameRate.s32SrcFrameRate = FRAME_RATE;
    stVpssChnAttr.stFrameRate.s32DstFrameRate = FRAME_RATE;
    stVpssChnAttr.u32Depth = 2;
    stVpssChnAttr.bMirror = CVI_FALSE;
    stVpssChnAttr.bFlip = CVI_FALSE;
    stVpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
    
    ret = CVI_VPSS_SetChnAttr(VPSS_GRP_ID, VPSS_CHN_ID, &stVpssChnAttr);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VPSS_SetChnAttr failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    // Step 3: Enable VPSS channel
    ret = CVI_VPSS_EnableChn(VPSS_GRP_ID, VPSS_CHN_ID);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VPSS_EnableChn failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    // Step 4: Start VPSS group
    ret = CVI_VPSS_StartGrp(VPSS_GRP_ID);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VPSS_StartGrp failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    std::cout << "VPSS system initialized successfully" << std::endl;
    return CVI_SUCCESS;
}

// Initialize VENC (Video Encoder) for JPEG with minimal direct API configuration
CVI_S32 initVideoEncoder() {
    std::cout << "Initializing VENC system with minimal configuration..." << std::endl;
    
    // Step 1: Setup minimal VENC channel attributes
    VENC_CHN_ATTR_S stVencChnAttr;
    memset(&stVencChnAttr, 0, sizeof(VENC_CHN_ATTR_S));
    
    // Basic VENC attributes - minimal setup
    stVencChnAttr.stVencAttr.enType = PT_JPEG;
    stVencChnAttr.stVencAttr.u32MaxPicWidth = IMAGE_WIDTH;
    stVencChnAttr.stVencAttr.u32MaxPicHeight = IMAGE_HEIGHT;
    stVencChnAttr.stVencAttr.u32PicWidth = IMAGE_WIDTH;
    stVencChnAttr.stVencAttr.u32PicHeight = IMAGE_HEIGHT;
    // Calculate buffer size and align to 1024 bytes (required by JPEG encoder)
    CVI_U32 u32BufSize = IMAGE_WIDTH * IMAGE_HEIGHT / 2;
    stVencChnAttr.stVencAttr.u32BufSize = (u32BufSize + 1023) & ~1023;  // Align to 1024
    stVencChnAttr.stVencAttr.bByFrame = CVI_TRUE;
    stVencChnAttr.stVencAttr.bEsBufQueueEn = CVI_FALSE;  // Disable queue for simplicity
    
    // JPEG specific attributes - minimal setup
    stVencChnAttr.stVencAttr.stAttrJpege.bSupportDCF = CVI_FALSE;
    stVencChnAttr.stVencAttr.stAttrJpege.stMPFCfg.u8LargeThumbNailNum = 0;
    stVencChnAttr.stVencAttr.stAttrJpege.enReceiveMode = VENC_PIC_RECEIVE_SINGLE;
    
    // GOP attributes - minimal for JPEG (single frame)
    stVencChnAttr.stGopAttr.enGopMode = VENC_GOPMODE_NORMALP;
    stVencChnAttr.stGopAttr.stNormalP.s32IPQpDelta = 0;
    
    // Rate control - use CBR mode for stability
    stVencChnAttr.stRcAttr.enRcMode = VENC_RC_MODE_MJPEGCBR;
    stVencChnAttr.stRcAttr.stMjpegCbr.u32StatTime = 1;
    stVencChnAttr.stRcAttr.stMjpegCbr.u32SrcFrameRate = FRAME_RATE;
    stVencChnAttr.stRcAttr.stMjpegCbr.fr32DstFrameRate = FRAME_RATE;
    stVencChnAttr.stRcAttr.stMjpegCbr.u32BitRate = 1000000;  // 1Mbps
    
    CVI_S32 ret = CVI_VENC_CreateChn(VENC_CHN_ID, &stVencChnAttr);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VENC_CreateChn failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    // Step 2: Start receiving frames
    VENC_RECV_PIC_PARAM_S stRecvParam;
    memset(&stRecvParam, 0, sizeof(VENC_RECV_PIC_PARAM_S));
    stRecvParam.s32RecvPicNum = -1; // Unlimited frames
    
    ret = CVI_VENC_StartRecvFrame(VENC_CHN_ID, &stRecvParam);
    if (ret != CVI_SUCCESS) {
        std::cerr << "CVI_VENC_StartRecvFrame failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    std::cout << "VENC system initialized successfully (minimal config)" << std::endl;
    return CVI_SUCCESS;
}

// Bind the complete pipeline: VI -> VPSS -> VENC
CVI_S32 bindVideoPipeline() {
    std::cout << "Binding video pipeline..." << std::endl;
    
    // Bind VI to VPSS
    MMF_CHN_S stSrcChn = {CVI_ID_VI, VI_PIPE_ID, VI_CHN_ID};
    MMF_CHN_S stDestChn = {CVI_ID_VPSS, VPSS_GRP_ID, 0};
    
    CVI_S32 ret = CVI_SYS_Bind(&stSrcChn, &stDestChn);
    if (ret != CVI_SUCCESS) {
        std::cerr << "VI->VPSS bind failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    // Bind VPSS to VENC
    stSrcChn = {CVI_ID_VPSS, VPSS_GRP_ID, VPSS_CHN_ID};
    stDestChn = {CVI_ID_VENC, 0, VENC_CHN_ID};
    
    ret = CVI_SYS_Bind(&stSrcChn, &stDestChn);
    if (ret != CVI_SUCCESS) {
        std::cerr << "VPSS->VENC bind failed: 0x" << std::hex << ret << std::endl;
        return ret;
    }
    
    std::cout << "Video pipeline bound successfully: VI->VPSS->VENC" << std::endl;
    return CVI_SUCCESS;
}

// Capture frames directly from VENC
void captureFrames() {
    std::cout << "Starting frame capture..." << std::endl;
    
    while (g_running.load()) {
        // Check time limit
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - g_start_time);
        if (elapsed.count() >= 5000) {
            g_running.store(false);
      break;
    }
        
        // Get VENC stream
        VENC_STREAM_S stStream;
        CVI_S32 ret = CVI_VENC_GetStream(VENC_CHN_ID, &stStream, 1000); // 1 second timeout
        
        if (ret == CVI_SUCCESS) {
            // Process the stream
            if (stStream.u32PackCount > 0) {
                int frame_num = g_frame_count.fetch_add(1) + 1;
                VENC_PACK_S* pstPack = &stStream.pstPack[0];
                
                uint8_t* jpeg_data = pstPack->pu8Addr + pstPack->u32Offset;
                uint32_t jpeg_size = pstPack->u32Len - pstPack->u32Offset;
                
                if (jpeg_data && jpeg_size > 0) {
                    std::string base64_str = macaron::Base64::Encode(
                        std::string(reinterpret_cast<char*>(jpeg_data), jpeg_size)
                    );
                    
                    std::cout << "Frame " << frame_num << " [" << elapsed.count() << "ms]" 
                             << " Size: " << jpeg_size << " bytes" << std::endl;
                    std::cout << "data:image/jpeg;base64," << base64_str << std::endl;
                    std::cout << std::endl;
                }
            }
            
            // Release the stream
            CVI_VENC_ReleaseStream(VENC_CHN_ID, &stStream);
        } else if (ret != CVI_ERR_VENC_BUSY) {
            std::cerr << "CVI_VENC_GetStream failed: 0x" << std::hex << ret << std::endl;
            usleep(100000); // 100ms delay on error
        }

        // If no frames after 1s, dump VI/VENC status for diagnostics
        if (g_frame_count.load() == 0 && elapsed.count() > 1000) {
            VI_CHN_STATUS_S viStat; memset(&viStat, 0, sizeof(VI_CHN_STATUS_S));
            if (CVI_VI_QueryChnStatus(VI_PIPE_ID, VI_CHN_ID, &viStat) == CVI_SUCCESS) {
                std::cerr << "VI Status - Width:" << viStat.stSize.u32Width
                          << " Height:" << viStat.stSize.u32Height << std::endl;
            }
            VENC_CHN_STATUS_S vencStat; memset(&vencStat, 0, sizeof(VENC_CHN_STATUS_S));
            if (CVI_VENC_QueryStatus(VENC_CHN_ID, &vencStat) == CVI_SUCCESS) {
                std::cerr << "VENC Status - LeftPics:" << vencStat.u32LeftPics << std::endl;
            }
        }
        
        usleep(33000); // ~30 FPS
    }
    
    std::cout << "Frame capture completed. Total frames: " << g_frame_count.load() << std::endl;
}

// Wrapper helpers
static inline bool startWrapperPipeline() {
    // Mirror example: ensure clean state, then init, setup, register, start with small delays
    deinitVideo();
    if (initVideo() != 0) {
        std::cerr << "initVideo failed" << std::endl;
        return false;
    }
    // Mirror example defaults: CH0 = H264 1920x1080@15, CH1 = JPEG 640x640@15 (preview)
    video_ch_param_t ch0{};
    ch0.format = VIDEO_FORMAT_H264;
    ch0.width = 1920;
    ch0.height = 1080;
    ch0.fps = 15;
    if (setupVideo(VIDEO_CH0, &ch0) != 0) {
        std::cerr << "setupVideo CH0 failed" << std::endl;
        return false;
    }

    video_ch_param_t ch1{};
    ch1.format = VIDEO_FORMAT_JPEG;
    ch1.width = 640;
    ch1.height = 480; // revert to the working 640x480 that produced frames
    ch1.fps = 15;
    if (setupVideo(VIDEO_CH1, &ch1) != 0) {
        std::cerr << "setupVideo CH1 failed" << std::endl;
        return false;
    }

    if (registerVideoFrameHandler(VIDEO_CH1, 0, [](void* pData, void* pArgs, void* pUserData)->int {
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
        std::cout << "Frame " << idx << " [" << elapsed << "ms] Size: " << len << " bytes\n";
        std::cout << "data:image/jpeg;base64," << b64 << "\n\n";
        return 0;
    }, nullptr) != 0) {
        std::cerr << "registerVideoFrameHandler failed" << std::endl;
        return false;
    }
    usleep(100000); // 100ms pre-start delay (example does small sleeps)
    if (startVideo() != 0) {
        std::cerr << "startVideo failed (continuing like earlier working state)" << std::endl;
        // Proceed; frames may still arrive via existing pipeline state
    }
    usleep(1000000); // 1s stabilization like example
    return true;
}

int main() {
    std::cout << "CVITEK ReCamera Frame Capture - Wrapper Pipeline" << std::endl;
    std::cout << "Will capture frames for 5 seconds and output as base64" << std::endl;
    
    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Ensure wrapper state is clean
    deinitVideo();
    
    // Start wrapper pipeline (working reference from example)
    if (!startWrapperPipeline()) {
        std::cerr << "ERROR: Failed to start wrapper pipeline" << std::endl;
        deinitVideo();
        return -1;
    }
    std::cout << std::endl << "=== WRAPPER PIPELINE READY ===" << std::endl;
    std::cout << "Channels: CH0 H264 1920x1080@15, CH1 JPEG 640x640@15" << std::endl;
    std::cout << "Starting 5-second capture..." << std::endl;
    std::cout << std::endl;
    
    // Warm-up and timed run (callback prints frames)
    usleep(2000000); // additional warm-up for AE/AWB
    g_start_time = std::chrono::steady_clock::now();
    usleep(5000000); // 5 seconds capture window
    
    std::cout << std::endl << "=== CAPTURE COMPLETED ===" << std::endl;
    
    // Cleanup everything
    std::cout << "Cleaning up video system..." << std::endl;
    deinitVideo();
    
    std::cout << "Program finished successfully." << std::endl;
  return 0;
}