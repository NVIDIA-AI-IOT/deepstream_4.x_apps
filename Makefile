################################################################################
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

APP:= deepstream-custom-uff

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=4.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

# Change to your deepstream 4.0 SDK includes
CFLAGS+= -I/Your_deepstream_SDK_v4.0_xxxxx_path/sources/includes

SRCS:= $(wildcard *.c)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0

OBJS:= $(SRCS:.c=.o)

CFLAGS+= `pkg-config --cflags $(PKGS)`

LIBS:= `pkg-config --libs $(PKGS)`

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
       -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(APP)


