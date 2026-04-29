# Makefile

NVCC      = nvcc
CXX_HOST  = /usr/bin/g++-11
EXTRA_FLAGS ?=
NVCCFLAGS = -O2 -ccbin $(CXX_HOST) -std=c++17 $(EXTRA_FLAGS)
LIBS      = -lcublas

# Set USE_NCCL=1 to enable tensor-parallel (multi-GPU) support
USE_NCCL ?= 0

# Source and build directories
SRC_DIR = src
BUILD_DIR = build

# Sources
SRCS = $(SRC_DIR)/main.cu \
       $(SRC_DIR)/forward_single.cu \
       $(SRC_DIR)/bench_common.cu \
       $(SRC_DIR)/bench_ar.cu \
       $(SRC_DIR)/bench_sd.cu \
       $(SRC_DIR)/bench_ssd.cu

# Headers (for dependency tracking)
HDRS = $(SRC_DIR)/config.h \
       $(SRC_DIR)/check.h \
       $(SRC_DIR)/gemm.h \
       $(SRC_DIR)/gpu_context.h \
       $(SRC_DIR)/forward_single.h \
       $(SRC_DIR)/bench_common.h \
       $(SRC_DIR)/bench_ar.h \
       $(SRC_DIR)/bench_sd.h \
       $(SRC_DIR)/bench_ssd.h

ifeq ($(USE_NCCL),1)
  NVCCFLAGS += -DUSE_NCCL
  LIBS      += -lnccl -lnvToolsExt
  SRCS      += $(SRC_DIR)/forward_tp.cu
  HDRS      += $(SRC_DIR)/forward_tp.h
endif

# Object files
OBJS = $(SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Binary
TARGET = benchmark

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

# Compile .cu -> .o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HDRS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean