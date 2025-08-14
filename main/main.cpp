#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <string>

#include <video.h>
#include <opencv2/opencv.hpp>
#include <porting/ma_osal.h>

#include <sscma.h>

using namespace ma;


#define CAMERA_INIT()                                                                                                                        \
    {                                                                                                                                        \
        Thread::enterCritical();                                                                                                             \
        Thread::sleep(Tick::fromMilliseconds(100));                                                                                          \
        MA_LOGI(TAG, "start video");                                                                                                         \
        startVideo();                                                                                                                        \
        Thread::sleep(Tick::fromSeconds(1));                                                                                                 \
        Thread::exitCritical();                                                                                                              \
        server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", "enabled"}, {"code", MA_OK}, {"data", enabled_.load()}})); \
    }

#define CAMERA_DEINIT()                                                                                                                      \
    {                                                                                                                                        \
        Thread::enterCritical();                                                                                                             \
        MA_LOGI(TAG, "stop video");                                                                                                          \
        Thread::sleep(Tick::fromMilliseconds(100));                                                                                          \
        deinitVideo();                                                                                                                       \
        Thread::sleep(Tick::fromSeconds(1));                                                                                                 \
        Thread::exitCritical();                                                                                                              \
        server_->response(id_, json::object({{"type", MA_MSG_TYPE_RESP}, {"name", "enabled"}, {"code", MA_OK}, {"data", enabled_.load()}})); \
    }

int vpssCallback(void* pData, void* pArgs) {
  APP_VENC_CHN_CFG_S* pstVencChnCfg = (APP_VENC_CHN_CFG_S*)pArgs;
  VIDEO_FRAME_INFO_S* VpssFrame     = (VIDEO_FRAME_INFO_S*)pData;
  VIDEO_FRAME_S* f                  = &VpssFrame->stVFrame;

  std::cerr << "vpssCallback" << std::endl;
  std::cerr.flush();
  
  auto channel = pstVencChnCfg->VencChn;

  // videoFrame* frame   = new videoFrame();
  // frame->chn          = pstVencChnCfg->VencChn;
  // frame->img.size     = f->u32Length[0] + f->u32Length[1] + f->u32Length[2];
  // frame->img.width    = channels_[pstVencChnCfg->VencChn].width;
  // frame->img.height   = channels_[pstVencChnCfg->VencChn].height;
  // frame->img.format   = channels_[pstVencChnCfg->VencChn].format;
  // frame->img.key      = true;
  // frame->img.physical = true;
  // frame->img.data     = reinterpret_cast<uint8_t*>(f->u64PhyAddr[0]);
  // frame->timestamp    = Tick::current();
  // frame->fps          = channels_[pstVencChnCfg->VencChn].fps;
  // frame->ref(channels_[pstVencChnCfg->VencChn].msgboxes.size());
  // for (auto& msgbox : channels_[pstVencChnCfg->VencChn].msgboxes) {
  //     if (!msgbox->post(frame, Tick::fromMilliseconds(5))) {
  //         frame->release();
  //     }
  // }
  return CVI_SUCCESS;
}

int vpssCallbackStub(void* pData, void* pArgs, void* pUserData) {
  APP_VENC_CHN_CFG_S* pstVencChnCfg = (APP_VENC_CHN_CFG_S*)pArgs;
  return vpssCallback(pData, pArgs);
}

int main() {
  std::cout << "Hello, ReCamera!" << std::endl;

  video_ch_param_t param;
  param.format = VIDEO_FORMAT_JPEG;
  param.width  = 1920;
  param.height = 1080;
  param.fps    = 10;

  std::cerr << "start channel 0 format " << param.format << " width " << param.width << " height " << param.height << " fps " << param.fps << std::endl;
  std::cerr.flush();

  initVideo();  // Initialize video parameters first
  setupVideo(VIDEO_CH0, &param);
  registerVideoFrameHandler(VIDEO_CH0, 0, vpssCallbackStub, NULL);

  Thread::enterCritical();
  Thread::sleep(Tick::fromMilliseconds(100));
  std::cerr << "start video" << std::endl;
  std::cerr.flush();
  startVideo();
  Thread::sleep(Tick::fromSeconds(1));
  Thread::exitCritical();
  std::cerr << "start video done!" << std::endl;
  std::cerr.flush();

  return 0;
}
