
# FasterRCNN, SSD, and MaskRCNN samples with Deepstream SDK

This repository provides 3 DeepStream sample apps based on [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk).

* **FasterRCNN sample** This sample shows how to use FasterRCNN model trained with [NVIDIA Transfer Learning Toolkit(TLT) SDK](https://developer.nvidia.com/transfer-learning-toolkit) to do inference with DeepStream SDK.
* **SSD sample** This sample shows how to use the SSD model trained with [NVIDIA Transfer Learning Toolkit(TLT) SDK](https://developer.nvidia.com/transfer-learning-toolkit) to do inference with DeepStream SDK.
* **MaskRCNN sample** This sample shows how to use the model trained with the [popular open sourced MaskRCNN implementation in GitHub](https://github.com/matterport/Mask_RCNN) to do inference with DeepStream SDK.

These DeepStream samples support both NVIDIA Tesla and Tegra platform.

The complete pipeline for these sample apps is:
> filesrc->h264parse->nvv4l2decoder->streammux->nvinfer(frcnn/ssd/mrcnn)->nvosd->nveglglesink

## Prerequisites

* [Deepstream SDK 4.0+](https://developer.nvidia.com/deepstream-sdk)
 You can run deepstream-test1 sample to check Deepstream installation is successful or not.

* [TensorRT 5.1 GA](https://developer.nvidia.com/tensorrt)

* [TensorRT OSS (release/5.1 branch)](https://github.com/NVIDIA/TensorRT/tree/release/5.1)
This repository depends on the TensorRT OSS plugins. Specifically, the FasterRCNN sample depends on the `cropAndResizePlugin` and `proposalPlugin`; the MaskRCNN sample depends on the `ProposalLayer_TRT`, `PyramidROIAlign_TRT`, `DetectionLayer_TRT` and `SpecialSlice_TRT`; the SSD sample depends on the `batchTilePlugin`. To use these plugins for the samples here, complile a new `libnvinfer_plugin.so*` and replace your system `libnvinfer_plugin.so*`.
Please note that TensorRT OSS 5.1 branch does not support cross compilation. To compile and replace the plugin,
 > $ git clone -b release/5.1 https://github.com/nvidia/TensorRT  && cd  TensorRT
 > $ git submodule update --init --recursive && export TRT_SOURCE=`pwd`
 > $ cd $TRT_SOURCE
 > $ mkdir -p build && cd build
 > $ wget https://github.com/Kitware/CMake/releases/download/v3.13.5/cmake-3.13.5.tar.gz
 > $ tar xvf cmake-3.13.5.tar.gz
 > $ cd cmake-3.13.5/ && ./configure && make && sudo make install
 > $ cd ..
 > $ /usr/local/bin/cmake .. -DTRT_BIN_DIR=`pwd`/out
 > $ make nvinfer_plugin -j$(nproc)
 > The libnvinfer_plugin.so* will be available in the `pwd`/out folder.  Then replace the system lib with the newly built lib.
 > $ sudo cp /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.5.x.x    /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.5.x.x.bak
 > $ sudo cp `pwd`/out/libnvinfer_plugin.so.5.x.x    /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.5.x.x


## Build
 * $ export DS_SRC_PATH="Your deepstream sdk source path".
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
  data_sources {
    tfrecords_path: "/home/projects2_metropolis/datasets/maglev_tfrecords/ivalarge_tfrecord_qres/*"
    image_directory_path: "/home/IVAData2/datasets/ivalarge_cyclops-b"
  }
  data_sources {
    tfrecords_path: "/home/projects2_metropolis/datasets/maglev_tfrecords/its_datasets_qres/aicities_highway/*"
    image_directory_path: "/home/projects2_metropolis/exports/IVA-0010-01_181016"
  }
  validation_fold: 0
  image_extension: "jpg"
  target_class_mapping {
    key: "AutoMobile"
    value: "car"
  }
  target_class_mapping {
    key: "Automobile"
    value: "car"
  }
  target_class_mapping {
    key: "Bicycle"
    value: "bicycle"
  }
  target_class_mapping {
    key: "Heavy Truck"
    value: "car"
  }
  target_class_mapping {
    key: "Motorcycle"
    value: "bicycle"
  }
  target_class_mapping {
    key: "Person"
    value: "person"
  }

  ...

  }
  target_class_mapping {
    key: "traffic_light"
    value: "road_sign"
  }
  target_class_mapping {
    key: "twowheeler"
    value: "bicycle"
  }
  target_class_mapping {
    key: "vehicle"
    value: "car"
  }
}
```

The corresponding label file will be

```
bicycle
car
person
road_sign
```


* **MaskRCNN**
TBD

### DeepStream configuration file
The DeepStream configuration file provides some parameters for DeepStream at runtime. For example, the model path, the label file path, the precision to run at for TensorRT backend, input and output node names, input dimensions, etc. For different apps, although most of the fields in the configuration file are similar, there are some minor differences. So we describe them one by one below.
Please refer to [DeepStream Development Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream_Development_Guide%2Fdeepstream_app_config.3.2.html) for detailed explanations of those parameters.

Once you finish training a model with Transfer Learning Toolkit, you can run `tlt-export` command to generate an `.etlt` model. This model can be deployed on DeepStream for fast inference. The DeepStream sample app can also accept the TensorRT engine(plan) file generated by running the `tlt-converter` tool on the `.etlt` model. The TensorRT engine file is hardware dependent, while the `.etlt` model is not. You may specify either a TensorRT engine file or a `.etlt` model in the config file, as below.

* **FasterRCNN**
  The FasterRCNN configuration file is `pgie_frcnn_uff_config.txt`. You may need some customization when you train your own model with TLT. A sample FasterRCNN configuration file looks like below. Each field is self-explanatory.

  A sample `.etlt` model is available at `models/frcnn/faster_rcnn.etlt`. The pb model under `models/frcnn` should not be used for this sample.

  ```
  [property]
  gpu-id=0
  net-scale-factor=1.0
  offsets=103.939;116.779;123.68
  model-color-format=1
  labelfile-path=./nvdsinfer_customparser_frcnn_uff/frcnn_labels.txt
  # Provide the .etlt model exported by TLT or a TensorRT engine created by tlt-converter
  # If use .etlt model, please also specify the key('nvidia_tlt')
  # model-engine-file=./faster_rcnn.uff_b1_fp32.engine
  tlt-encoded-model=./models/frcnn/faster_rcnn.etlt
  tlt-model-key=nvidia_tlt
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
Make sure "deepstream-test1" sample can run before running this app.
Once we have built the app and finished the configuration, we can run the app, using the command mentioned below.
```bash
./deepstream-custom <config_file> <H264_file>
```

## Known issues and Notes
* To run FasterRCNN/SSD in fp16 mode, please replace "/opt/nvidia/deepstream/deepstream-4.0/lib/libnvds_inferutils.so" in your platform by `fp16_fix/libnvds_inferutils.so.aarch64` or `fp16_fix/libnvds_inferutils.so.x86`

* For SSD, don't forget to set your own keep_count, keep_top_k in `nvdsinfer_custombboxparser_ssd_uff.cpp` for the NMS layer, if you change them in the training stage in TLT.

* For FasterRCNN, don't forget to set your own parameters in `nvdsinfer_customparser_frcnn_uff/nvdsinfer_customparser_frcnn_uff.cpp` if you change them in the training stage in TLT.

* For MaskRCNN, app can show bbox but cannot show mask in present. User can dump mask in the buffer `out_mask` in `nvdsinfer_customparser_mrcnn_uff/nvdsinfer_custombboxparser_mrcnn_uff.cpp`.

* In function 'attach_metadata_detector()' in deepstream source code:
 1. frame scale_ratio_x/scale_ratio_y is (network width/height) / (streammux width/height)
 2. Some objects will be filtered because its width/height/top/left is beyond the source size (streammux is as source)
