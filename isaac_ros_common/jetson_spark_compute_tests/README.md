# Compute Sanity Tests

These are simple tests designed for basic debugging and sanity checking of Isaac ROS compute packages on Jetson and DGX Spark systems.

The primary purpose of these tests is to confirm that the core functionalities of the compute packages are operational. This serves as a quick verification that the Isaac ROS development container environment is configured and working as expected. The intent is to use these for debugging if package tests are failing.

## Available Tests

### 1. TensorRT Function Test (`trt_function_test`)
Tests basic TensorRT functionality.

**Usage:**
```bash
make trt_function_test
./trt_function_test
```

### 2. V4L2 Compatibility Test (`test_v4l2_compat`)
Tests CUDA and V4L2 encoder compatibility on DGX Spark systems. This test verifies that CUDA initialization and V4L2 encoder operations work correctly together by performing the complete encoder configuration sequence.

**Purpose:**
- Verifies CUDA library compatibility with V4L2 hardware encoder
- Diagnoses CUDA + V4L2 initialization issues on DGX Spark
- Tests the full encoder setup sequence used by Isaac ROS compression nodes

**Usage:**
```bash
make test_v4l2_compat
LD_PRELOAD=/opt/deepstream_tegra/lib/libv4l2.so.0 ./test_v4l2_compat
```

**Expected Output:**
- On success: All steps pass and "RESULT: PASSED" is displayed
- On failure: Specific step failure is shown with error details

**Common Failure Points:**
- `v4l2_open()`: CUDA libraries incompatible with GPU
- `VIDIOC_S_PARM`: Most common failure point indicating library incompatibility
