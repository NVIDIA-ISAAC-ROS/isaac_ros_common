CXX = g++
TARGET = trt_function_test
SRC = trt_function_test.cpp

# Compiler and Linker Flags
CPPFLAGS = -I/usr/include/aarch64-linux-gnu \
	   -I/usr/local/cuda/include
LDFLAGS = -L/usr/lib/aarch64-linux-gnu \
	  -L/usr/local/cuda/lib64
LIBS = -lnvinfer \
       -lnvonnxparser \
       -lpthread \
       -ldl \
       -lcudart

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CPPFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(LIBS)

# Clean target
clean:
	rm -f $(TARGET) *.o

.PHONY: all clean



