# ailia-models

The collection of pre-trained, state-of-the-art models.

## About ailia SDK

ailia SDK is a cross-platform high speed inference SDK. The ailia SDK provides a consistent C++ API on Windows, Mac, Linux, iOS, Android and Jetson. It supports Unity, Python and JNI for efficient AI implementation. The ailia SDK makes great use of the GPU via Vulkan and Metal to serve accelerated computing.

You can download a free evaluation version that allows you to evaluate the ailia SDK for 30 days. Please download from the trial link below.

https://ailia.jp/en/

## Image classification

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [vgg16](/vgg16/) |[Very Deep Convolutional Networks for Large-Scale Image Recognition]( https://arxiv.org/abs/1409.1556 )|Keras| 1.1.0 and later|
| [googlenet](/googlenet/) |[Going Deeper with Convolutions]( https://arxiv.org/abs/1409.4842 )|Pytorch| 1.2.0 and later|
| [resnet50](/resnet50/) | [Deep Residual Learning for Image Recognition]( https://github.com/KaimingHe/deep-residual-networks) | Chainer | 1.2.0 and later |
| [inceptionv3](/inceptionv3/)|[Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)|Pytorch| 1.2.0 and later |
| [mobilenetv2](/mobilenetv2/)|[PyTorch Implemention of MobileNet V2](https://github.com/d-li14/mobilenetv2.pytorch)|Pytorch| 1.2.0 and later |
| [mobilenetv3](/mobilenetv3/)|[PyTorch Implemention of MobileNet V3](https://github.com/d-li14/mobilenetv3.pytorch)|Pytorch| 1.2.1 and later |
| [partialconv](/partialconv/)|[Partial Convolution Layer for Padding and Image Inpainting](https://github.com/NVIDIA/partialconv)|Pytorch| 1.2.0 and later |
| [efficientnet](/efficientnet/)|[A PyTorch implementation of EfficientNet]( https://github.com/lukemelas/EfficientNet-PyTorch)|Pytorch| 1.2.3 beta and later |

## Image segmentation

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [deeplabv3](/deeplabv3/) | [Xception65 for backbone network of DeepLab v3+](https://github.com/tensorflow/models/tree/master/research/deeplab) | Chainer | 1.2.0 and later |
| [hrnet_segmentation](/hrnet_segmentation/) | [High-resolution networks (HRNets) for Semantic Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) | Pytorch | 1.2.1 and later |
| [hair_segmentation](/hair_segmentation/) | [hair segmentation in mobile device](https://github.com/thangtran480/hair-segmentation) | Keras | 1.2.1 and later |
| [pspnet-hair-segmentation](/pspnet-hair-segmentation/) | [pytorch-hair-segmentation](https://github.com/YBIGTA/pytorch-hair-segmentation) | Pytorch | 1.2.2 and later |

## Image manipulation

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [srresnet](/srresnet/) | [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://github.com/twtygqyy/pytorch-SRResNet) | Pytorch | 1.2.0 and later |
| [noise2noise](/noise2noise/) | [Learning Image Restoration without Clean Data](https://github.com/joeylitalien/noise2noise-pytorch) | Pytorch | 1.2.0 and later |
| [dewarpnet](/dewarpnet) | [DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks](https://github.com/cvlab-stonybrook/DewarpNet) | Pytorch | 1.2.1 and later |

## Object detection

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [yolov1-tiny](/yolov1-tiny/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolov1/) | Darknet | 1.1.0 and later |
| [yolov1-face](/yolov1-face/) | [YOLO-Face-detection](https://github.com/dannyblueliu/YOLO-Face-detection/) | Darknet | 1.1.0 and later |
| [yolov2](/yolov2/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | Pytorch | 1.2.0 and later |
| [yolov3](/yolov3/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | ONNX Runtime | 1.2.1 and later |
| [yolov3-tiny](/yolov3-tiny/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | ONNX Runtime | 1.2.1 and later |
| [yolov3-face](/yolov3-face/) | [Face detection using keras-yolov3](https://github.com/axinc-ai/yolov3-face) | Keras | 1.2.1 and later |
| [yolov3-hand](/yolov3-hand/) | [Hand detection branch of Face detection using keras-yolov3](https://github.com/axinc-ai/yolov3-face/tree/hand_detection) | Keras | 1.2.1 and later |
| [mobilenet_ssd](/mobilenet_ssd/) | [MobileNetV1, MobileNetV2, VGG based SSD/SSD-lite implementation in Pytorch](https://github.com/qfgaohao/pytorch-ssd) | Pytorch | 1.2.1 and later |
| [maskrcnn](/maskrcnn/) | [Mask R-CNN: real-time neural network for object instance segmentation](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/mask-rcnn) | Pytorch | 1.2.3 beta and later |

## Pose estimation

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[openpose](/openpose/) | [Code repo for realtime multi-person pose estimation in CVPR'17 (Oral)](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) | Caffe | 1.2.1 and later |
|[lightweight-human-pose-estimation](/lightweight-human-pose-estimation/) | [Fast and accurate human pose estimation in PyTorch. Contains implementation of "Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose" paper.](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) | Pytorch | 1.2.1 and later |
|[lightweight-human-pose-estimation-3d](/lightweight-human-pose-estimation-3d/) | [Real-time 3D multi-person pose estimation demo in PyTorch. OpenVINO backend can be used for fast inference on CPU.](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch) | Pytorch | 1.2.1 and later |


## Gaze estimation

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [gazeml](/gazeml/) | [A deep learning framework based on Tensorflow for the training of high performance gaze estimation](https://github.com/swook/GazeML) | TensorFlow | 1.2.0 and later |

## Face recognization

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[face_classification](/face_classification) | [Real-time face detection and emotion/gender classification](https://github.com/oarriaga/face_classification) | Keras | 1.1.0 and later |
|[vggface2](/vggface2) | [VGGFace2 Dataset for Face Recognition](https://github.com/ox-vgg/vgg_face2) | Caffe | 1.1.0 and later |
|[facial_feature](/facial_feature/)|[kaggle-facial-keypoints](https://github.com/axinc-ai/kaggle-facial-keypoints)|Pytorch| 1.2.0 and later |
|[face_alignment](/face_alignment/)| [2D and 3D Face alignment library build using pytorch](https://github.com/1adrianb/face-alignment) | Pytorch | 1.2.1 and later |
|[blazeface](/blazeface/)| [BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch) | Pytorch | 1.2.1 and later |
|[arcface](/arcface/) | [pytorch implement of arcface](https://github.com/ronghuaiyang/arcface-pytorch) | Pytorch | 1.2.1 and later |

## Rotation estimation

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[rotnet](/rotnet) | [CNNs for predicting the rotation angle of an image to correct its orientation](https://github.com/d4nst/RotNet) | Keras | 1.2.1 and later |

## Crowd counting

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[crowdcount-cascaded-mtl](/crowdcount-cascaded-mtl) | [CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting (Single Image Crowd Counting)](https://github.com/svishwa/crowdcount-cascaded-mtl) | Pytorch | 1.2.1 and later |

## Depth estimation

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[monodepth2](monodepth2)| [Monocular depth estimation from a single image](https://github.com/nianticlabs/monodepth2) | Pytorch | 1.2.2 and later |

## Text recognization

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[etl](/etl) | Japanese Character Recognization | Keras | 1.1.0 and later |

## Natural language processing

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[bert](/bert) | [pytorch-pretrained-bert](https://pypi.org/project/pytorch-pretrained-bert/) | Pytorch | 1.2.2 and later |
