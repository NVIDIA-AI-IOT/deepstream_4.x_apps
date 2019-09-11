/**
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"
#include <cassert>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

// Max number of final detections
static const int DETECTION_MAX_INSTANCES = 100;

// Number of classification classes (including background)
static const int NUM_CLASSES = 1 + 80; // COCO has 80 classes

static const int MASK_POOL_SIZE = 14;
static const nvinfer1::DimsCHW INPUT_SHAPE{3, 1024, 1024};
//static const Dims2 MODEL_DETECTION_SHAPE{DETECTION_MAX_INSTANCES, 6};
//static const Dims4 MODEL_MASK_SHAPE{DETECTION_MAX_INSTANCES, NUM_CLASSES, 28, 28};

struct MRCNNBBox {
    float x1, y1, x2, y2;
};

struct MRCNNMask {
    float raw[MASK_POOL_SIZE * 2 * MASK_POOL_SIZE * 2];
};

struct MRCNNBBoxInfo {
    MRCNNBBox box;
    int label = -1;
    float prob = 0.0f;

    MRCNNMask* mask = nullptr;
};

struct RawDetection {
    float y1, x1, y2, x2, class_id, score;
};


/* This is a sample bounding box parsing function for the sample mask rcnn
 *
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomMrcnnUff (
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo  const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

//static int64_t volume(const nvinfer1::Dims& d) {
//    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
//}

std::vector<MRCNNBBoxInfo> decodeOutput(void* detectionsHost, void* masksHost) {
    int input_dim_h = INPUT_SHAPE.d[1];
    int input_dim_w = INPUT_SHAPE.d[2];
    assert(input_dim_h == input_dim_w);

    std::vector<MRCNNBBoxInfo> binfo;

    //int detectionOffset = volume(MODEL_DETECTION_SHAPE); // (100,6)
    //int maskOffset = volume(MODEL_MASK_SHAPE);           // (100, 81, 28, 28)

    RawDetection* detections
        = reinterpret_cast<RawDetection*>((float*) detectionsHost);
    MRCNNMask* masks = reinterpret_cast<MRCNNMask*>((float*) masksHost);
    for (int det_id = 0; det_id < DETECTION_MAX_INSTANCES; det_id++) {
        RawDetection cur_det = detections[det_id];
        int label = (int) cur_det.class_id;
        if (label <= 0)
            continue;

        MRCNNBBoxInfo det;
        det.label = label;
        det.prob = cur_det.score;

        det.box.x1 = cur_det.x1 ;
        det.box.y1 = cur_det.y1 ;
        det.box.x2 = cur_det.x2 ;
        det.box.y2 = cur_det.y2 ;

        if (det.box.x2 <= det.box.x1 || det.box.y2 <= det.box.y1)
            continue;

        det.mask = masks + det_id * NUM_CLASSES + label;

        binfo.push_back(det);
    }

    return binfo;
}

extern "C"
bool NvDsInferParseCustomMrcnnUff (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {

    static int detIndex = -1;
    static int maskIndex = -1;

    /* Find the detection layer */
    if (detIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
            if (strcmp(outputLayersInfo[i].layerName, "mrcnn_detection") == 0) {
                detIndex = i;
                break;
            }
        }
        if (detIndex == -1) {
            std::cerr << "Could not find detection layer buffer while parsing" << std::endl;
            return false;
        }
    }

    /* Find the mask layer */
    if (maskIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
            if (strcmp(outputLayersInfo[i].layerName, "mrcnn_mask/Sigmoid") == 0) {
                maskIndex = i;
                break;
            }
        }
        if (maskIndex == -1) {
            std::cerr << "Could not find mask layer buffer while parsing" << std::endl;
            return false;
        }
    }

    float* out_det = (float *) outputLayersInfo[detIndex].buffer;
    float* out_mask = (float *) outputLayersInfo[maskIndex].buffer;

    std::vector<MRCNNBBoxInfo> binfo = decodeOutput(out_det, out_mask);
    for (unsigned int roi_id = 0; roi_id < binfo.size(); roi_id++) {
        NvDsInferObjectDetectionInfo object;
        object.classId = binfo[roi_id].label;
        object.detectionConfidence = binfo[roi_id].prob;

        /* Clip object box co-ordinates to network resolution */
        object.left = CLIP(binfo[roi_id].box.x1 * networkInfo.width, 0, networkInfo.width - 1);
        object.top = CLIP(binfo[roi_id].box.y1 * networkInfo.height, 0, networkInfo.height - 1);
        object.width = CLIP((binfo[roi_id].box.x2 - binfo[roi_id].box.x1) * networkInfo.width, 0, networkInfo.width - 1);
        object.height = CLIP((binfo[roi_id].box.y2 - binfo[roi_id].box.y1) * networkInfo.height, 0, networkInfo.height - 1);

        objectList.push_back(object);

    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomMrcnnUff);
