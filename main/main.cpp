#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <string>

#include <signal.h>
#include <video.h>
#include <opencv2/opencv.hpp>
#include <porting/ma_osal.h>
#include <sscma.h>

#include "base64.hpp"

using namespace ma;

bool gotFrame = false;

int vpssCallback(void* pData, void* pArgs) {
  APP_VENC_CHN_CFG_S* pstVencChnCfg = (APP_VENC_CHN_CFG_S*)pArgs;
  VIDEO_FRAME_INFO_S* VpssFrame     = (VIDEO_FRAME_INFO_S*)pData;
  VIDEO_FRAME_S* f                  = &VpssFrame->stVFrame;
  ma_img_t img;

  std::cerr << std::endl << std::endl << "== vpssCallback ==" << std::endl;
  std::cerr.flush();

  auto channel = pstVencChnCfg->VencChn;
  std::cerr << "channel " << channel << std::endl;
  std::cerr.flush();

  memset(&img, 0, sizeof(img));
  img.size     = f->u32Length[0] + f->u32Length[1] + f->u32Length[2];
  img.width    = 1920;
  img.height   = 1080;
  img.format   = MA_PIXEL_FORMAT_JPEG;
  img.key      = true;
  img.physical = true;
  img.data     = reinterpret_cast<uint8_t*>(f->pu8VirAddr[0]);
  img.timestamp = Tick::current();
  img.index     = 0;
  img.count     = 1;
  img.rotate    = MA_PIXEL_ROTATE_0;

  std::cerr << "img.size: " << img.size << std::endl;
  std::cerr << "img.width: " << img.width << std::endl;
  std::cerr << "img.height: " << img.height << std::endl;
  std::cerr << "img.format: " << img.format << std::endl;
  std::cerr << "img.key: " << img.key << std::endl;
  std::cerr << "img.physical: " << img.physical << std::endl;
  std::cerr << "img.timestamp: " << img.timestamp << std::endl;
  std::cerr << "img.index: " << img.index << std::endl;
  std::cerr << "img.count: " << img.count << std::endl;
  std::cerr << "img.rotate: " << img.rotate << std::endl;
  std::cerr.flush();

  // print jpg as base64
  std::string base64 = macaron::Base64::Encode(std::string(reinterpret_cast<char*>(f->pu8VirAddr[0]), f->u32Length[0]));
  std::cerr << "data:image/jpeg;base64," << base64 << std::endl;
  std::cerr.flush();

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

  gotFrame = true;
  return CVI_SUCCESS;
}

int vpssCallbackStub(void* pData, void* pArgs, void* pUserData) {
  APP_VENC_CHN_CFG_S* pstVencChnCfg = (APP_VENC_CHN_CFG_S*)pArgs;
  return vpssCallback(pData, pArgs);
}

static void initializeVideoSystem() {
  Thread::enterCritical();
  Thread::sleep(Tick::fromMilliseconds(100));
  std::cerr << "start video" << std::endl;
  std::cerr.flush();
  startVideo();
  Thread::sleep(Tick::fromSeconds(1));
  Thread::exitCritical();
  std::cerr << "start video done!" << std::endl;
  std::cerr.flush();
}

static void deinitializeVideoSystem() {
  Thread::enterCritical();
  std::cerr << "stop video" << std::endl;
  Thread::sleep(Tick::fromMilliseconds(100));
  deinitVideo();
  Thread::sleep(Tick::fromSeconds(1));
  Thread::exitCritical();
  std::cerr << "stop video done!" << std::endl;
  std::cerr.flush();
}

int main() {
  std::cout << "Hello, ReCamera!" << std::endl;

  if (initVideo() != 0) {
    std::cerr << "initVideo failed" << std::endl;
    std::cerr.flush();
    return -1;
  }

  Signal::install({SIGINT, SIGSEGV, SIGABRT, SIGTRAP, SIGTERM, SIGHUP, SIGQUIT, SIGPIPE}, [](int sig) {
    std::cerr << "received signal " << sig << std::endl;
    std::cerr.flush();
    deinitializeVideoSystem();
    exit(1);
  });

  video_ch_param_t param;
  param.format = VIDEO_FORMAT_JPEG;
  param.width  = 1920;
  param.height = 1080;
  param.fps    = 10;

  std::cerr << "start channel 0 format " << param.format << " width " << param.width << " height " << param.height << " fps " << param.fps << std::endl;
  std::cerr.flush();

  setupVideo(VIDEO_CH0, &param);
  registerVideoFrameHandler(VIDEO_CH0, 0, vpssCallbackStub, NULL);
  initializeVideoSystem();

  while (true) {
    Thread::sleep(Tick::fromMilliseconds(1));
    if (gotFrame) {
      std::cerr << "got frame" << std::endl;
      std::cerr.flush();
      break;
    }
  }

  // free the camera
  deinitializeVideoSystem();

  return 0;
}
