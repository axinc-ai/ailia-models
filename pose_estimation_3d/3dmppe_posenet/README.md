# PoseNet of "Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image"

### input
![input image](input.jpg)

(from https://github.com/mks0601/3DMPPE_POSENET_RELEASE/tree/master/demo)

### output
![output_image](output.png)

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python 3dmppe_posenet.py
(ex on CPU)  $ python 3dmppe_posenet.py -e 0
(ex on BLAS) $ python 3dmppe_posenet.py -e 1
(ex on GPU)  $ python 3dmppe_posenet.py -e 2
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 3dmppe_posenet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
$ python3 3dmppe_posenet.py -i IMAGE_PATH -s SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.
```bash
$ python3 3dmppe_posenet.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
$ python3 3dmppe_posenet.py -v VIDEO_PATH -s SAVE_VIDEO_PATH
(ex) $ python3 3dmppe_posenet.py --video input.mp4 --savepath output.mp4
```

### Reference

[Mask R-CNN : real-time neural network for object instance segmentation](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/mask-rcnn)
[PoseNet of "Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image"](https://github.com/mks0601/3DMPPE_POSENET_RELEASE)


### Framework
Pytorch

### Model Format
ONNX opset = 10


### Netron

[mask_rcnn_R_50_FPN_1x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mask_rcnn/mask_rcnn_R_50_FPN_1x.onnx.prototxt)
[rootnet_snapshot_18.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/3dmppe_posenet/rootnet_snapshot_18.opt.onnx.prototxt)
[posenet_snapshot_24.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/3dmppe_posenet/posenet_snapshot_24.opt.onnx.prototxt)
