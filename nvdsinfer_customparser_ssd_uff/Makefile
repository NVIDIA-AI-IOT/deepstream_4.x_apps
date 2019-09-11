###############################################################################
#
# Copyright (c) 2018-2019 NVIDIA CORPORATION.  All Rights Reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
###############################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
CC:= g++

# Change to your deepstream 4.0 SDK includes
CFLAGS+= -I/Your_deepstream_SDK_v4.0_xxxxx_path/sources/includes \
	  -I/usr/local/cuda-$(CUDA_VER)/include

CFLAGS+= -Wall -std=c++11 -shared -fPIC

LIBS+= -lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas

LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsinfer_custombboxparser_ssd_uff.cpp
TARGET_LIB:= libnvds_infercustomparser_ssd_uff.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
