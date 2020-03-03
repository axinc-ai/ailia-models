# ailia-models

The collection of pre-trained, state-of-the-art models.

## About ailia SDK

ailia SDK is a cross-platform high speed inference SDK. The ailia SDK provides a consistent C++ API on Windows, Mac, Linux, iOS and Android. It supports Unity, Python and JNI for efficient AI implementation. The ailia SDK makes great use of the GPU to serve accelerated computing.

https://ailia.jp/en/

## Image classification

| Name | Detail | Exported From | Supported Version |
|:-----------|------------:|:------------:|:------------:|
| [vgg16](/vgg16/) |[Very Deep Convolutional Networks for Large-Scale Image Recognition]( https://arxiv.org/abs/1409.1556 )|Keras| 1.1.0|
| [googlenet](/googlenet/) |[Going Deeper with Convolutions]( https://arxiv.org/abs/1409.4842 )|Pytorch| 1.2.0|
| [resnet50](/resnet50/) | [Deep Residual Learning for Image Recognition]( https://github.com/KaimingHe/deep-residual-networks) | Chainer | 1.2.0 |
| [inceptionv3](/inceptionv3/)|[Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)|Pytorch| 1.2.0 |
| [mobilenetv2](/mobilenetv2/)|[PyTorch Implemention of MobileNet V2](https://github.com/d-li14/mobilenetv2.pytorch)|Pytorch| 1.2.0 |
| [mobilenetv3](/mobilenetv3/)|[PyTorch Implemention of MobileNet V3](https://github.com/d-li14/mobilenetv3.pytorch)|Pytorch| 1.2.1 |
| [partialconv](/partialconv/)|[Partial Convolution Layer for Padding and Image Inpainting](https://github.com/NVIDIA/partialconv)|Pytorch| 1.2.0 |

## Image segmentation

| Name | Detail | Exported From | Supported Version |
|:-----------|------------:|:------------:|:------------:|
| [deeplabv3](/deeplabv3/) | [Xception65 for backbone network of DeepLab v3+](https://github.com/tensorflow/models/tree/master/research/deeplab) | Chainer | 1.2.0  |
| [hrnet_segmentation](/hrnet_segmentation/) | [High-resolution networks (HRNets) for Semantic Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) | Pytorch | 1.2.1 |
| [hair_segmentation](/hair_segmentation/) | [hair segmentation in mobile device](https://github.com/thangtran480/hair-segmentation) | Keras | 1.2.1 |

## Image manipulation

| Name | Detail | Exported From | Supported Version |
|:-----------|------------:|:------------:|:------------:|
| [srresnet](/srresnet/) | [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://github.com/twtygqyy/pytorch-SRResNet) | Pytorch | 1.2.0 |
| [noise2noise](/noise2noise/) | [Learning Image Restoration without Clean Data](https://github.com/joeylitalien/noise2noise-pytorch) | Pytorch | 1.2.0 |
| [dewarpnet](/dewarpnet) | [DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks](https://github.com/cvlab-stonybrook/DewarpNet) | Pytorch | 1.2.1 |

## Object detection

| Name | Detail | Exported From | Supported Version |
|:-----------|------------:|:------------:|:------------:|
| [yolov1-tiny](/yolov1-tiny/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolov1/) | Darknet | 1.1.0  |
| [yolov1-face](/yolov1-face/) | [YOLO-Face-detection](https://github.com/dannyblueliu/YOLO-Face-detection/) | Darknet | 1.1.0  |
| [yolov2](/yolov2/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | Pytorch | 1.2.0  |
| [yolov3](/yolov3/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | ONNX Runtime | 1.2.1  |
| [yolov3-tiny](/yolov3-tiny/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | Keras | 1.2.1  |
| [yolov3-face](/yolov3-face/) | [Face detection using keras-yolov3](https://github.com/axinc-ai/yolov3-face) | Keras | 1.2.1  |
| [mobilenet_ssd](/mobilenet_ssd/) | [MobileNetV1, MobileNetV2, VGG based SSD/SSD-lite implementation in Pytorch](https://github.com/qfgaohao/pytorch-ssd) | Pytorch | 1.2.1  |

## Pose estimation

| Name | Detail | Exported From | Supported Version |
|:-----------|------------:|:------------:|:------------:|
|[openpose](/openpose/) | [Code repo for realtime multi-person pose estimation in CVPR'17 (Oral)](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) | Caffe | 1.2.1 |
|[lightweight-human-pose-estimation](/lightweight-human-pose-estimation/) | [Fast and accurate human pose estimation in PyTorch. Contains implementation of "Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose" paper.](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) | Pytorch | 1.2.1 |

## Gaze estimation

| Name | Detail | Exported From | Supported Version |
|:-----------|------------:|:------------:|:------------:|
| [gazeml](/gazeml/) | [A deep learning framework based on Tensorflow for the training of high performance gaze estimation](https://github.com/swook/GazeML) | TensorFlow | 1.2.0 |

## Face recognization

| Name | Detail | Exported From | Supported Version |
|:-----------|------------:|:------------:|:------------:|
|[face_classification](/face_classification) | [Real-time face detection and emotion/gender classification](https://github.com/oarriaga/face_classification) | Keras | 1.1.0 |
|[vggface2](/vggface2) | [VGGFace2 Dataset for Face Recognition](https://github.com/ox-vgg/vgg_face2) | Caffe | 1.1.0 |
|[facial_feature](/facial_feature/)|[kaggle-facial-keypoints](https://github.com/axinc-ai/kaggle-facial-keypoints)|Pytorch| 1.2.0 |
|[face_alignment](/face_alignment/)| [2D and 3D Face alignment library build using pytorch](https://github.com/1adrianb/face-alignment) | Pytorch | 1.2.1 |

## Rotation estimation

| Name | Detail | Exported From | Supported Version |
|:-----------|------------:|:------------:|:------------:|
|[rotnet](/rotnet) | [CNNs for predicting the rotation angle of an image to correct its orientation](https://github.com/d4nst/RotNet) | Keras | 1.2.1 |

## Crowd counting

| Name | Detail | Exported From | Supported Version |
|:-----------|------------:|:------------:|:------------:|
|[crowdcount-cascaded-mtl](/crowdcount-cascaded-mtl) | [CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting (Single Image Crowd Counting)](https://github.com/svishwa/crowdcount-cascaded-mtl) | Pytorch | 1.2.1 |

## Text recognization

| Name | Detail | Exported From | Supported Version |
|:-----------|------------:|:------------:|:------------:|
|[etl](/etl) | Japanese Character Recognization | Keras | 1.1.0 |
