CXX = g++
TARGETS = trt_function_test test_v4l2_compat
SRC_TRT = trt_function_test.cpp
SRC_V4L2 = test_v4l2_compat.cpp

# Compiler and Linker Flags
CPPFLAGS = -I/usr/include/aarch64-linux-gnu \
	   -I/usr/local/cuda/include
LDFLAGS = -L/usr/lib/aarch64-linux-gnu \
	  -L/usr/local/cuda/lib64 \
	  -L/usr/lib/aarch64-linux-gnu/nvidia
LIBS = -lnvinfer \
       -lnvonnxparser \
       -lpthread \
       -ldl \
       -lcudart

# V4L2 compatibility test libraries
V4L2_LIBS = -lcudart -lv4l2 -ldl

# Default target
all: $(TARGETS)

trt_function_test: $(SRC_TRT)
	$(CXX) $(CPPFLAGS) $(SRC_TRT) -o trt_function_test $(LDFLAGS) $(LIBS)

test_v4l2_compat: $(SRC_V4L2)
	$(CXX) $(CPPFLAGS) $(SRC_V4L2) -o test_v4l2_compat $(LDFLAGS) $(V4L2_LIBS)

# Clean target
clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean



