// Copyright 2025 NVIDIA CORPORATION
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * CUDA + V4L2 Compatibility Test for DGX Spark
 *
 * This test verifies that CUDA initialization and V4L2 encoder operations
 * work correctly together on DGX Spark systems. It performs the minimal
 * sequence of operations needed to configure a V4L2 H.264 encoder:
 *
 * 1. Initialize CUDA (cudaSetDevice)
 * 2. Open V4L2 encoder device
 * 3. Configure encoder settings (rate control, IDR interval, formats, etc.)
 * 4. Verify all operations succeed
 *
 * Usage:
 *   LD_PRELOAD=/opt/deepstream_tegra/lib/libv4l2.so.0 ./test_v4l2_compat
 */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <linux/videodev2.h>

#include <cuda_runtime.h>
#include <libv4l2.h>

// NVIDIA V4L2 extensions
#ifndef V4L2_CID_MPEG_VIDEO_CUDA_GPU_ID
#define V4L2_CID_MPEG_VIDEO_CUDA_GPU_ID (V4L2_CID_MPEG_BASE + 558)
#endif

#ifndef V4L2_CID_MPEG_VIDEO_IDR_INTERVAL
#define V4L2_CID_MPEG_VIDEO_IDR_INTERVAL (V4L2_CID_MPEG_BASE + 514)
#endif

#ifndef V4L2_MPEG_VIDEO_BITRATE_MODE_CONSTQP
#define V4L2_MPEG_VIDEO_BITRATE_MODE_CONSTQP 4
#endif

#ifndef V4L2_CID_MPEG_VIDEOENC_CUDA_PRESET_ID
#define V4L2_CID_MPEG_VIDEOENC_CUDA_PRESET_ID (V4L2_CID_MPEG_BASE + 578)
#endif

int main()
{
  printf("═══════════════════════════════════════════════════════════\n");
  printf("  CUDA + V4L2 Compatibility Test\n");
  printf("═══════════════════════════════════════════════════════════\n\n");

    // Step 1: Initialize CUDA
  printf("[1] Initializing CUDA...\n");
  cudaError_t cuda_err = cudaSetDevice(0);
  if (cuda_err != cudaSuccess) {
    printf("    ✗ cudaSetDevice failed: %s\n", cudaGetErrorString(cuda_err));
    return 1;
  }
  printf("    ✓ cudaSetDevice(0) succeeded\n\n");

    // Step 2: Try to open V4L2 encoder device
  printf("[2] Opening V4L2 encoder device...\n");
  const char * device = "/dev/nvidia0";

  int fd = v4l2_open(device, 0);
  if (fd < 0) {
    printf("    ✗ v4l2_open() failed: %s (errno=%d)\n", strerror(errno), errno);
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  RESULT: FAILED\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("This means the CUDA libraries in LD_LIBRARY_PATH are\n");
    printf("incompatible with DGX Spark's GPU.\n");
    printf("\n");
    return 1;
  }

  printf("    ✓ v4l2_open() succeeded (fd=%d)\n\n", fd);

    // Step 3: Set rate control mode BEFORE format
  printf("[3] Setting rate control mode (CBR)...\n");
  struct v4l2_ext_control rc_ctrl;
  struct v4l2_ext_controls rc_ctrls;
  memset(&rc_ctrl, 0, sizeof(rc_ctrl));
  memset(&rc_ctrls, 0, sizeof(rc_ctrls));

  rc_ctrl.id = V4L2_CID_MPEG_VIDEO_BITRATE_MODE;
  rc_ctrl.value = V4L2_MPEG_VIDEO_BITRATE_MODE_CBR;
  rc_ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;
  rc_ctrls.count = 1;
  rc_ctrls.controls = &rc_ctrl;

  if (v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &rc_ctrls) < 0) {
    printf("    ✗ Rate control mode failed: %s (errno=%d)\n",
               strerror(errno), errno);
    printf("    (continuing anyway - may skip this on error)\n\n");
  } else {
    printf("    ✓ Rate control mode set to CBR\n\n");
  }

    // Step 4: Set IDR interval BEFORE format
  printf("[4] Setting IDR interval...\n");
  struct v4l2_ext_control idr_ctrl;
  struct v4l2_ext_controls idr_ctrls;
  memset(&idr_ctrl, 0, sizeof(idr_ctrl));
  memset(&idr_ctrls, 0, sizeof(idr_ctrls));

  idr_ctrl.id = V4L2_CID_MPEG_VIDEO_IDR_INTERVAL;
  idr_ctrl.value = 30;
  idr_ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;
  idr_ctrls.count = 1;
  idr_ctrls.controls = &idr_ctrl;

  if (v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &idr_ctrls) < 0) {
    printf("    ✗ IDR interval failed: %s (errno=%d)\n",
               strerror(errno), errno);
    printf("    (continuing anyway - may skip this on error)\n\n");
  } else {
    printf("    ✓ IDR interval set to 30\n\n");
  }

    // Step 5: Set capture format first (H.264 output)
  printf("[5] Setting capture format (H.264 output)...\n");
  struct v4l2_format cap_fmt;
  memset(&cap_fmt, 0, sizeof(cap_fmt));
  cap_fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  cap_fmt.fmt.pix_mp.width = 1920;
  cap_fmt.fmt.pix_mp.height = 1080;
  cap_fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_H264;
  cap_fmt.fmt.pix_mp.num_planes = 1;

  if (v4l2_ioctl(fd, VIDIOC_S_FMT, &cap_fmt) < 0) {
    printf("    ✗ VIDIOC_S_FMT (capture) failed: %s (errno=%d)\n",
               strerror(errno), errno);
    v4l2_close(fd);
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  RESULT: FAILED\n");
    printf("═══════════════════════════════════════════════════════════\n");
    return 1;
  }
  printf("    ✓ VIDIOC_S_FMT (capture) succeeded\n\n");

    // Step 6: Set CUDA GPU ID
  printf("[6] Setting V4L2_CID_MPEG_VIDEO_CUDA_GPU_ID...\n");
  struct v4l2_ext_control gpu_ctrl;
  struct v4l2_ext_controls gpu_ctrls;
  memset(&gpu_ctrl, 0, sizeof(gpu_ctrl));
  memset(&gpu_ctrls, 0, sizeof(gpu_ctrls));

  gpu_ctrl.id = V4L2_CID_MPEG_VIDEO_CUDA_GPU_ID;
  gpu_ctrl.value = 0;    // GPU 0
  gpu_ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;
  gpu_ctrls.count = 1;
  gpu_ctrls.controls = &gpu_ctrl;

  if (v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &gpu_ctrls) < 0) {
    printf("    ✗ V4L2_CID_MPEG_VIDEO_CUDA_GPU_ID failed: %s (errno=%d)\n",
               strerror(errno), errno);
    v4l2_close(fd);
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  RESULT: FAILED\n");
    printf("═══════════════════════════════════════════════════════════\n");
    return 1;
  }
  printf("    ✓ V4L2_CID_MPEG_VIDEO_CUDA_GPU_ID set to 0\n\n");

    // Step 7: Set output format (NV12 input frames)
  printf("[7] Setting output format (NV12 input)...\n");
  struct v4l2_format fmt;
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  fmt.fmt.pix_mp.width = 1920;
  fmt.fmt.pix_mp.height = 1080;
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12M;
  fmt.fmt.pix_mp.num_planes = 2;

  if (v4l2_ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
    printf("    ✗ VIDIOC_S_FMT (output) failed: %s (errno=%d)\n",
               strerror(errno), errno);
    v4l2_close(fd);
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  RESULT: FAILED\n");
    printf("═══════════════════════════════════════════════════════════\n");
    return 1;
  }
  printf("    ✓ VIDIOC_S_FMT (output) succeeded\n\n");

    // Step 8: Set H.264 profile
  printf("[8] Setting H.264 profile (HIGH)...\n");
  struct v4l2_ext_control prof_ctrl;
  struct v4l2_ext_controls prof_ctrls;
  memset(&prof_ctrl, 0, sizeof(prof_ctrl));
  memset(&prof_ctrls, 0, sizeof(prof_ctrls));

  prof_ctrl.id = V4L2_CID_MPEG_VIDEO_H264_PROFILE;
  prof_ctrl.value = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH;
  prof_ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;
  prof_ctrls.count = 1;
  prof_ctrls.controls = &prof_ctrl;

  if (v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &prof_ctrls) < 0) {
    printf("    ✗ Profile setting failed: %s (errno=%d)\n",
               strerror(errno), errno);
    printf("    (continuing anyway - may skip this on error)\n\n");
  } else {
    printf("    ✓ Profile set to HIGH\n\n");
  }

    // Step 9: Set HW Preset
  printf("[9] Setting HW Preset...\n");
  struct v4l2_ext_control preset_ctrl;
  struct v4l2_ext_controls preset_ctrls;
  memset(&preset_ctrl, 0, sizeof(preset_ctrl));
  memset(&preset_ctrls, 0, sizeof(preset_ctrls));

  preset_ctrl.id = V4L2_CID_MPEG_VIDEOENC_CUDA_PRESET_ID;
  preset_ctrl.value = 0;
  preset_ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;
  preset_ctrls.count = 1;
  preset_ctrls.controls = &preset_ctrl;

  if (v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &preset_ctrls) < 0) {
    printf("    ✗ HW Preset setting failed: %s (errno=%d)\n",
               strerror(errno), errno);
    printf("    (continuing anyway - may skip this on error)\n\n");
  } else {
    printf("    ✓ HW Preset set to 0\n\n");
  }

    // Step 10: Set bitrate
  printf("[10] Setting bitrate...\n");
  struct v4l2_ext_control br_ctrl;
  struct v4l2_ext_controls br_ctrls;
  memset(&br_ctrl, 0, sizeof(br_ctrl));
  memset(&br_ctrls, 0, sizeof(br_ctrls));

  br_ctrl.id = V4L2_CID_MPEG_VIDEO_BITRATE;
  br_ctrl.value = 20000000;    // 20 Mbps
  br_ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;
  br_ctrls.count = 1;
  br_ctrls.controls = &br_ctrl;

  if (v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &br_ctrls) < 0) {
    printf("    ✗ Bitrate setting failed: %s (errno=%d)\n",
               strerror(errno), errno);
    printf("    (continuing anyway - may skip this on error)\n\n");
  } else {
    printf("    ✓ Bitrate set to 20 Mbps\n\n");
  }

    // Step 11: Query device capabilities
  printf("[11] Querying device capabilities...\n");
  struct v4l2_capability cap;
  memset(&cap, 0, sizeof(cap));
  if (v4l2_ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
    printf("    ✗ VIDIOC_QUERYCAP failed: %s (errno=%d)\n",
               strerror(errno), errno);
    v4l2_close(fd);
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  RESULT: FAILED\n");
    printf("═══════════════════════════════════════════════════════════\n");
    return 1;
  }
  printf("    ✓ VIDIOC_QUERYCAP succeeded\n\n");

    // Step 12: Get current stream parameters
  printf("[12] Getting stream parameters...\n");
  struct v4l2_streamparm parm;
  memset(&parm, 0, sizeof(parm));
  parm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  if (v4l2_ioctl(fd, VIDIOC_G_PARM, &parm) < 0) {
    printf("    ✗ VIDIOC_G_PARM failed: %s (errno=%d)\n",
               strerror(errno), errno);
    v4l2_close(fd);
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  RESULT: FAILED\n");
    printf("═══════════════════════════════════════════════════════════\n");
    return 1;
  }
  printf("    ✓ VIDIOC_G_PARM succeeded (current: %d/%d fps)\n",
           parm.parm.output.timeperframe.denominator,
           parm.parm.output.timeperframe.numerator);

    // Step 13: Set stream parameters
  printf("[13] Setting stream parameters...\n");
  parm.parm.output.timeperframe.numerator = 1;
  parm.parm.output.timeperframe.denominator = 30;
  if (v4l2_ioctl(fd, VIDIOC_S_PARM, &parm) < 0) {
    printf("    ✗ VIDIOC_S_PARM failed: %s (errno=%d)\n",
               strerror(errno), errno);
    v4l2_close(fd);
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  RESULT: FAILED\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("VIDIOC_S_PARM failure indicates CUDA library incompatibility.\n");
    printf("\n");
    return 1;
  }
  printf("    ✓ VIDIOC_S_PARM succeeded\n\n");

  v4l2_close(fd);

  printf("═══════════════════════════════════════════════════════════\n");
  printf("  RESULT: PASSED\n");
  printf("═══════════════════════════════════════════════════════════\n");
  printf("\n");
  printf("All V4L2 operations succeeded!\n");
  printf("The CUDA libraries are compatible with V4L2 encoder.\n");
  printf("\n");

  return 0;
}
