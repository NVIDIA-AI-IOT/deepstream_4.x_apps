
# FasterRCNN, SSD, and MaskRCNN sample with Deepstream 4.0

This repository provides 3 DeepStream sample apps based on [NVIDIA DeepStream 4.0 SDK](https://developer.nvidia.com/deepstream-sdk).

* **FasterRCNN sample** This sample shows how to use FasterRCNN model trained with [NVIDIA Transfer Learning Toolkit(TLT) SDK](https://developer.nvidia.com/transfer-learning-toolkit) to do inference with DeepStream 4.0.
* **SSD sample** This sample shows how to use the SSD model trained with [NVIDIA Transfer Learning Toolkit(TLT) SDK](https://developer.nvidia.com/transfer-learning-toolkit) to do inference with DeepStream 4.0.
* **MaskRCNN sample** This sample shows how to use the model trained with the [popular open sourced MaskRCNN implementation in GitHub](https://github.com/matterport/Mask_RCNN) to do inference with DeepStream 4.0.

These DeepStream samples support both NVIDIA Tesla and Tegra platform.

The complete pipeline for these sample apps is:
> filesrc->h264parse->nvv4l2decoder->streammux->nvinfer(frcnn/ssd/mrcnn)->nvosd->nveglglesink

## Prequisites

* [Deepstream 4.0](https://developer.nvidia.com/deepstream-sdk)

* [TensorRT 5.1 GA](https://developer.nvidia.com/tensorrt)

* [TensorRT OSS (release/5.1 branch)](https://github.com/NVIDIA/TensorRT)
This repository depends on the TensorRT OSS plugins. Specifically, the FasterRCNN sample depends on the `cropAndResizePlugin` and `proposalPlugin`; the MaskRCNN sample depends on the `ProposalLayer_TRT`, `PyramidROIAlign_TRT`, `DetectionLayer_TRT` and `SpecialSlice_TRT`; the SSD sample depends on the `batchTilePlugin`. To use these plugins for the samples here, complile a new `libnvinfer_plugin.so*/libnvinfer_plugin.a*` and replace your system `libnvinfer_plugin.so*/libnvinfer_plugin.a*`.

## Build
 * Replace `/Your_deepstream_SDK_v4.0_xxxxx_path` with your actual DeepStream SDK 4.0 path in `nvdsinfer_customparser_xxx_uff/Makefile` and `Makefile`.
 * $ cd nvdsinfer_customparser_frcnn_uff or nvdsinfer_customparser_ssd_uff or nvdsinfer_customparser_mrcnn_uff
 * $ make
 * $ cd ..
 * $ make

## Configure
We need to do some configurations before we can run these sample apps. The configuration includes two parts. One is the label file for the DNN model and the other is the DeepStream configuration file.

### Label file
The label provides the list of class names for a specific DNN model trained in one of the methods mentioned above. The label varies for different apps. Details given below.

* **FasterRCNN**
For FasterRCNN, the label file is `nvdsinfer_customparser_frcnn_uff/frcnn_labels.txt`. When training the FasterRCNN model you should have an experiment specification file. The labels can be found there. For example, suppose the `class_mapping` field in experiment specification file looks like

```
class_mapping {
key: 'Car'
value: 0
}
class_mapping {
key: 'Van'
value: 0
}
class_mapping {
key: "Pedestrian"
value: 1
}
class_mapping {
key: "Person_sitting"
value: 1
}
class_mapping {
key: 'Cyclist'
value: 2
}
class_mapping {
key: "background"
value: 3
}
class_mapping {
key: "DontCare"
value: -1
}
class_mapping {
key: "Truck"
value: -1
}
class_mapping {
key: "Misc"
value: -1
}
class_mapping {
key: "Tram"
value: -1
}
```
We choose an arbitrary key for each number if there are more than one key that maps to the same number. And we only include the keys that map to non-negative numbers since classes mapped to negative numbers are don't-care classes. Thus, the corresponding label file would be(has to be in the same order of the numbers in the class mapping):
```
Car
Pedestrian
Cyclist
background
```

* **SSD**
The order in which the classes are listed here must match the order in which the model predicts the output. This order is derived from the order in which the objects are instantiated in the `dataset_config` field of the SSD experiment config file as mentioned in Transfer Learning Toolkit user guide. For example, if the `dataset_config` is like this:

```
dataset_config {
  data_sources: {
    tfrecords_path: "/workspace/tlt-experiments/tfrecords/pascal_voc/pascal_voc*"
    image_directory_path: "/workspace/tlt-experiments/data/VOCdevkit/VOC2012"
  }
  image_extension: "jpg"
  target_class_mapping {
    key: "car"
    value: "car"
  }
  target_class_mapping {
    key: "person"
    value: "person"
  }
  target_class_mapping {
    key: "bicycle"
    value: "bicycle"
  }
  validation_fold: 0
}
```

The corresponding label file will be

```
car
person
bicycle
```


* **MaskRCNN**
TBD

### DeepStream configuration file
The DeepStream configuration file provides some parameters for DeepStream at runtime. For example, the model path, the label file path, the precision to run at for TensorRT backend, input and output node names, input dimensions, etc. For different apps, most of the fields in the configuration file are similar, although there are some differences. So we describe them one by one below.
Please refer to DeepStream user guide for detailed explanations of those arguments.

Once you finish training a model with Transfer Learning Toolkit, you can run `tlt-export` command to generate an `etlt` model. This model can also be deployed on DeepStream for fast inference.

* **FasterRCNN**
The FasterRCNN configuration file is `pgie_frcnn_uff_config.txt`. You may need some customization when you train your own model with TLT. A sample FasterRCNN configuration file looks like below. Each field is self-explanatory.
```
[property]
gpu-id=0
net-scale-factor=1.0
offsets=103.939;116.779;123.68
model-color-format=1
labelfile-path=./nvdsinfer_customparser_frcnn_uff/frcnn_labels.txt
# model can be either a uff model or an etlt model, but do not provide both
uff-file=./faster_rcnn.uff
model-engine-file=./faster_rcnn.uff_b1_fp32.engine
# tlt-encoded-model=./faster_rcnn.etlt
# tlt-model-key=<your_tlt_model_key>
uff-input-dims=3;272;480;0
uff-input-blob-name=input_1
batch-size=1
## 0=FP32, 1=INT8, 2=FP16 mode
network-mode=0
num-detected-classes=5
interval=0
gie-unique-id=1
is-classifier=0
#network-type=0
output-blob-names=dense_regress/BiasAdd;dense_class/Softmax;proposal
parse-bbox-func-name=NvDsInferParseCustomFrcnnUff
custom-lib-path=./nvdsinfer_customparser_frcnn_uff/libnvds_infercustomparser_frcnn_uff.so

[class-attrs-all]
roi-top-offset=0
roi-bottom-offset=0
detected-min-w=0
detected-min-h=0
detected-max-w=0
detected-max-h=0
```

* **SSD**
The SSD configuration file is `pgie_ssd_uff_config.txt`.

* **MaskRCNN**
The MaskRCNN configuration file is `pgie_mrcnn_uff_config.txt`.

## Run the sample app
Once we have built the app and finished the configuration, we can run the app, as below.
```bash
./deepstream-custom-uff <config_file> <H264_file>
```

## Known issues and Notes
* For frcnn fp16 mode, it needs the below patch and rebuild libnvds_infer.so
```
--- a/src/utils/nvdsinfer/nvdsinfer_context_impl.cpp
+++ b/src/utils/nvdsinfer/nvdsinfer_context_impl.cpp
@@ -1851,7 +1851,7 @@ NvDsInferContextImpl::generateTRTModel(
         }

         if (!uffParser->parse(initParams.uffFilePath,
-                    *network, modelDataType))
+                    *network, DataType::kFLOAT))
```

* Apply the below patch to `converter_functions.py` in UFF python package in TensorRT 5.1GA release to fix a bug about softmax layer in UFF converter(convert pb model to UFF model). This is only required for the FasterRCNN pb model provided here. If you use TLT to train your own model, then no action is required because the TLT container already done this for you.
```
--- converter_functions.py	2019-07-30 14:02:10.215925898 +0800
+++ converter_functions_fix_softmax.py	2019-07-30 13:53:46.187910972 +0800
@@ -231,7 +231,7 @@
     else:
         axis = 0
     fmt = convert_to_str(tf_node.attr['data_format'].s)
-    fmt = fmt if fmt else "NHWC"
+    fmt = fmt if fmt else "NCHW"
     data_fmt = tf2uff.convert_tf2uff_data_format(fmt)
     uff_graph.softmax(inputs[0], axis, data_fmt, name)
     return [tf2uff.split_node_name_and_output(inp)[0] for inp in inputs]

```

* For SSD, don't forget to set your own keep_count, keep_top_k in nvdsinfer_custombboxparser_ssd_uff.cpp for NMS layer if you change them in the training stage in TLT.

* For FasterRCNN, don't forget to set your own parameters in `nvdsinfer_customparser_frcnn_uff/nvdsinfer_customparser_frcnn_uff.cpp` if you change them in the training stage in TLT.

* For MaskRCNN, app can show bbox but cannot show mask in present. User can dump mask in the buffer `out_mask` in `nvdsinfer_customparser_mrcnn_uff/nvdsinfer_custombboxparser_mrcnn_uff.cpp`.

* In function 'attach_metadata_detector()' in deepstream source code:
 1. frame scale_ratio_x/scale_ratio_y is (network width/height) / (streammux width/height)
 2. Some objs will be filtered because its width/height/top/left is beyond the source size (streammux is as source)
