<img src="ailia-models.png">

The collection of pre-trained, state-of-the-art AI models.

# About ailia SDK

[ailia SDK](https://ailia.jp/en/) is a cross-platform high speed inference SDK. The ailia SDK provides a consistent C++ API on Windows, Mac, Linux, iOS, Android, Jetson and Raspberry Pi. It supports Unity, Python and JNI for efficient AI implementation. The ailia SDK makes great use of the GPU via Vulkan and Metal to serve accelerated computing.

# How to use

[ailia MODELS tutorial](TUTORIAL.md)

# Supported models

## Action recognition

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="action_recognition/mars/inputs/input0.jpg" width=128px>](action_recognition/mars/) | [mars](/action_recognition/mars/) | [MARS: Motion-Augmented RGB Stream for Action Recognition](https://github.com/craston/MARS) | Pytorch | 1.2.4 and later | [EN](https://medium.com/axinc-ai/mars-a-machine-learning-model-for-identifying-actions-from-videos-6b93c06ac6a5) [JP](https://medium.com/axinc/mars-%E5%8B%95%E7%94%BB%E3%81%8B%E3%82%89%E3%82%A2%E3%82%AF%E3%82%B7%E3%83%A7%E3%83%B3%E3%82%92%E8%AD%98%E5%88%A5%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-c03b0b8804a8) |
| [<img src="action_recognition/st_gcn/input.png" width=128px>](action_recognition/st_gcn/) | [st-gcn](/action_recognition/st_gcn/) | [ST-GCN](https://github.com/yysijie/st-gcn) | Pytorch | 1.2.5 and later | [EN](https://medium.com/axinc-ai/st-gcn-a-machine-learning-model-for-detecting-human-actions-from-skeletons-46a95b31b5db) [JP](https://medium.com/axinc/st-gcn-%E9%AA%A8%E6%A0%BC%E3%81%8B%E3%82%89%E4%BA%BA%E7%89%A9%E3%81%AE%E3%82%A2%E3%82%AF%E3%82%B7%E3%83%A7%E3%83%B3%E3%82%92%E6%A4%9C%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-af3196e38d1f) |
| [<img src="action_recognition/ax_action_recognition/punch_03-12-09-21-27-876_out.gif" width=128px>](action_recognition/ax_action_recognition/) | [ax_action_recognition](/action_recognition/ax_action_recognition/) | [Realtime-Action-Recognition](https://github.com/felixchenfy/Realtime-Action-Recognition) | Pytorch | 1.2.7 and later | |
| [<img src="action_recognition/va-cnn/image/f-0.png" width=128px>](action_recognition/va-cnn/) | [va-cnn](/action_recognition/va-cnn/) | [View Adaptive Neural Networks (VA) for Skeleton-based Human Action Recognition](https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition) | Pytorch | 1.2.7 and later ||
| [<img src="action_recognition/driver-action-recognition-adas/img/out_030.png" width=128px>](action_recognition/driver-action-recognition-adas/) | [driver-action-recognition-adas](/action_recognition/driver-action-recognition-adas/) | [driver-action-recognition-adas-0002](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/driver-action-recognition-adas-0002) | OpenVINO | 1.2.5 and later | |

## Anomaly detection

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="anomaly_detection/padim/output.png" width=128px>](anomaly_detection/padim/) | [padim](/anomaly_detection/padim/) | [PaDiM-Anomaly-Detection-Localization-master](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master) | Pytorch | 1.2.6 and later | [EN](https://medium.com/axinc-ai/padim-a-machine-learning-model-for-detecting-defective-products-without-retraining-5daa6f203377) [JP](https://medium.com/axinc/padim-%E5%86%8D%E5%AD%A6%E7%BF%92%E4%B8%8D%E8%A6%81%E3%81%A7%E4%B8%8D%E8%89%AF%E5%93%81%E6%A4%9C%E7%9F%A5%E3%82%92%E8%A1%8C%E3%81%86%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-69add653fbd3) |

## Audio processing

| Model | Reference | Exported From | Supported Ailia Version | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|
| [crnn_audio_classification](/audio_processing/crnn_audio_classification/) | [crnn-audio-classification](https://github.com/ksanjeevan/crnn-audio-classification) | Pytorch | 1.2.5 and later | [EN](https://medium.com/axinc-ai/crnnsoundclassification-a-machine-learning-model-for-classifying-sound-8e45d1f22fa) [JP](https://medium.com/axinc/crnnsoundclassification-%E9%9F%B3%E5%A3%B0%E3%82%92%E5%88%86%E9%A1%9E%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-2a35564dad42) |
| [deepspeech2](/audio_processing/deepspeech2/) | [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) | Pytorch | 1.2.2 and later |  [EN](https://medium.com/axinc-ai/deepspeech2-a-machine-learning-model-for-speech-recognition-d9e64c0d1afc) [JP](https://medium.com/axinc/deepspeech2-%E9%9F%B3%E5%A3%B0%E8%AA%8D%E8%AD%98%E3%82%92%E8%A1%8C%E3%81%86%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-8288c5f53eea) |
| [pytorch-dc-tts](/audio_processing/pytorch-dc-tts/) | [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://github.com/tugstugi/pytorch-dc-tts)   | Pytorch | 1.2.6 and later | [EN](https://medium.com/axinc-ai/pytorchdctts-a-machine-learning-model-for-text-to-speech-synthesis-2273e269b480) [JP](https://medium.com/axinc/pytorchdctts-%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%81%8B%E3%82%89%E9%9F%B3%E5%A3%B0%E5%90%88%E6%88%90%E3%82%92%E8%A1%8C%E3%81%86%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-dcb5eb07c883) |
| [unet_source_separation](/audio_processing/unet_source_separation/) | [source_separation](https://github.com/AppleHolic/source_separation)  | Pytorch | 1.2.6 and later | [EN](https://medium.com/axinc-ai/unetsourceseparation-a-machine-learning-model-to-remove-audio-noise-and-extract-voices-5acae8c37291) [JP](https://medium.com/axinc/unetsourceseparation-%E9%9B%91%E9%9F%B3%E3%82%92%E9%99%A4%E5%8E%BB%E3%81%97%E3%81%A6%E5%A3%B0%E3%81%A0%E3%81%91%E3%82%92%E6%8A%BD%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-5d23fd054eac) |
| [transformer-cnn-emotion-recognition](/audio_processing/transformer-cnn-emotion-recognition/) | [Combining Spatial and Temporal Feature Representions of Speech Emotion by Parallelizing CNNs and Transformer-Encoders](https://github.com/IliaZenkov/transformer-cnn-emotion-recognition)  | Pytorch | 1.2.5 and later |
| [auto_speech](/audio_processing/auto_speech/) | [AutoSpeech: Neural Architecture Search for Speaker Recognition](https://github.com/VITA-Group/AutoSpeech)  | Pytorch | 1.2.5 and later | [EN](https://medium.com/axinc-ai/autospeech-speech-based-person-identification-model-f01822f6d8e5) [JP](https://medium.com/axinc/autospeech-%E9%9F%B3%E5%A3%B0%E3%81%AB%E3%82%88%E3%82%8B%E5%80%8B%E4%BA%BA%E8%AD%98%E5%88%A5%E3%83%A2%E3%83%87%E3%83%AB-267a00f26a4a) |

## Background removal

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="background_removal/u2net/output.png" width=128px>](background_removal/u2net/) | [U-2-Net](/background_removal/u2net/) | [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/NathanUA/U-2-Net) | Pytorch | 1.2.2 and later | [EN](https://medium.com/axinc-ai/u2net-a-machine-learning-model-that-performs-object-cropping-in-a-single-shot-48adfc158483) [JP](https://medium.com/axinc/u2net-%E3%82%B7%E3%83%B3%E3%82%B0%E3%83%AB%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%E3%81%A7%E7%89%A9%E4%BD%93%E3%81%AE%E5%88%87%E3%82%8A%E6%8A%9C%E3%81%8D%E3%82%92%E8%A1%8C%E3%81%86%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-e346f2787cdb) |
| [<img src="background_removal/u2net-portrait-matting/output.png" width=128px>](background_removal/u2net-portrait-matting/) | [u2net-portrait-matting](/background_removal/u2net-portrait-matting/) | [U^2-Net - Portrait matting](https://github.com/dennisbappert/u-2-net-portrait) | Pytorch | 1.2.7 and later |
| [<img src="background_removal/u2net-human-seg/output.png" width=128px>](background_removal/u2net-human-seg/) | [u2net-human-seg](/background_removal/u2net-human-seg/) | [U^2-Net - human segmentation](https://github.com/xuebinqin/U-2-Net) | Pytorch | 1.2.4 and later |
| [<img src="background_removal/deep-image-matting/output.png" width=128px>](background_removal/deep-image-matting/) | [deep-image-matting](/background_removal/deep-image-matting/) | [Deep Image Matting](https://github.com/foamliu/Deep-Image-Matting)| Keras | 1.2.3 and later | [EN](https://medium.com/axinc-ai/deep-image-matting-a-machine-learning-model-to-improve-the-accuracy-of-image-matting-2ff98e0b47d6) [JP](https://medium.com/axinc/deep-image-matting-%E7%89%A9%E4%BD%93%E3%81%AE%E5%88%87%E3%82%8A%E6%8A%9C%E3%81%8D%E3%82%92%E9%AB%98%E7%B2%BE%E5%BA%A6%E5%8C%96%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-45580882966f) |
| [<img src="background_removal/indexnet/output.png" width=128px>](background_removal/indexnet/) | [indexnet](/background_removal/indexnet/) | [Indices Matter: Learning to Index for Deep Image Matting](https://github.com/open-mmlab/mmediting/tree/master/configs/mattors/indexnet) | Pytorch | 1.2.7 and later |
| [<img src="background_removal/modnet/output.jpg" width=128px>](background_removal/modnet/) | [modnet](/background_removal/modnet/) | [MODNet: Trimap-Free Portrait Matting in Real Time](https://github.com/ZHKKKe/MODNet) | Pytorch | 1.2.7 and later |
| [<img src="background_removal/background_matting_v2/output.png" width=128px>](background_removal/background_matting_v2/) | [background_matting_v2](/background_removal/background_matting_v2/) | [Real-Time High-Resolution Background Matting](https://github.com/PeterL1n/BackgroundMattingV2) | Pytorch | 1.2.9 and later |

## Crowd counting

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="crowd_counting/crowdcount-cascaded-mtl/result.png" width=256px>](crowd_counting/crowdcount-cascaded-mtl/) | [crowdcount-cascaded-mtl](/crowd_counting/crowdcount-cascaded-mtl) | [CNN-based Cascaded Multi-task Learning of <br/>High-level Prior and Density Estimation for Crowd Counting <br/>(Single Image Crowd Counting)](https://github.com/svishwa/crowdcount-cascaded-mtl) | Pytorch | 1.2.1 and later | [EN](https://medium.com/axinc-ai/crowdcounting-a-machine-learning-model-for-counting-people-a7b274a7c2af) [JP](https://medium.com/axinc/crowdcounting-%E7%94%BB%E5%83%8F%E3%81%8B%E3%82%89%E4%BA%BA%E6%95%B0%E3%82%92%E8%A8%88%E6%B8%AC%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-459e8b3fc184) |
| [<img src="crowd_counting/c-3-framework/output.png" width=256px>](crowd_counting/c-3-framework/) | [c-3-framework](/crowd_counting/c-3-framework) | [Crowd Counting Code Framework(C^3-Framework)](https://github.com/gjy3035/C-3-Framework) | Pytorch | 1.2.5 and later | |

## Deep fashion

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
|[<img src="deep_fashion/clothing-detection/output_modanet.png" width=128px>](deep_fashion/clothing-detection/) | [clothing-detection](/deep_fashion/clothing-detection/) | [Clothing-Detection](https://github.com/simaiden/Clothing-Detection) | Pytorch | 1.2.1 and later | [EN](https://medium.com/axinc-ai/clothingdetection-a-machine-learning-model-for-detecting-clothing-dab99e1492eb) [JP](https://medium.com/axinc/clothingdetection-%E6%9C%8D%E8%A3%85%E3%82%92%E6%A4%9C%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-e75cc8bc75b7) |
|[<img src="deep_fashion/mmfashion/output.png" width=128px>](deep_fashion/mmfashion/) | [mmfashion](/deep_fashion/mmfashion/) | [MMFashion](https://github.com/open-mmlab/mmfashion) | Pytorch | 1.2.5 and later | [EN](https://medium.com/axinc-ai/mmfashion-a-machine-learning-model-for-fashion-segmentation-a043fa972a2a) [JP](https://medium.com/axinc/mmfashion-%E3%83%95%E3%82%A1%E3%83%83%E3%82%B7%E3%83%A7%E3%83%B3%E3%82%92%E3%82%BB%E3%82%B0%E3%83%A1%E3%83%B3%E3%83%86%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-c486af72fdb5) |
|[<img src="deep_fashion/mmfashion_tryon/output.png" width=128px>](deep_fashion/mmfashion_tryon/) | [mmfashion_tryon](/deep_fashion/mmfashion_tryon/) | [MMFashion virtual try-on](https://github.com/open-mmlab/mmfashion) | Pytorch | 1.2.8 and later |
|[<img src="deep_fashion/mmfashion_retrieval/output.png" width=128px>](deep_fashion/mmfashion_retrieval/) | [mmfashion_retrieval](/deep_fashion/mmfashion_retrieval/) | [MMFashion In-Shop Clothes Retrieval](https://github.com/open-mmlab/mmfashion) | Pytorch | 1.2.5 and later |
|[<img src="deep_fashion/fashionai-key-points-detection/output_blouse.png" width=128px>](deep_fashion/fashionai-key-points-detection/) | [fashionai-key-points-detection](/deep_fashion/fashionai-key-points-detection/) | [A Pytorch Implementation of Cascaded Pyramid Network for FashionAI Key Points Detection](https://github.com/gathierry/FashionAI-KeyPointsDetectionOfApparel) | Pytorch | 1.2.5 and later |

## Depth estimation

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="depth_estimation/monodepth2/output.png" width=256px>](depth_estimation/monodepth2/) | [monodepth2](depth_estimation/monodepth2)| [Monocular depth estimation from a single image](https://github.com/nianticlabs/monodepth2) | Pytorch | 1.2.2 and later |
| [<img src="depth_estimation/midas/input_depth.png" width=256px>](depth_estimation/midas/) |[midas](depth_estimation/midas)| [Towards Robust Monocular Depth Estimation:<br/> Mixing Datasets for Zero-shot Cross-dataset Transfer](https://github.com/intel-isl/MiDaS) | Pytorch | 1.2.4 and later | [EN](https://medium.com/axinc-ai/midas-a-machine-learning-model-for-depth-estimation-e96119cc1a3c) [JP](https://medium.com/axinc/midas-%E5%A5%A5%E8%A1%8C%E3%81%8D%E3%82%92%E6%8E%A8%E5%AE%9A%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-71e65a041e0f) |
| [<img src="depth_estimation/fcrn-depthprediction/input_depth.png" width=256px>](depth_estimation/fcrn-depthprediction/) |[fcrn-depthprediction](depth_estimation/fcrn-depthprediction)| [Deeper Depth Prediction with Fully Convolutional Residual Networks](https://github.com/iro-cp/FCRN-DepthPrediction) | TensorFlow | 1.2.6 and later |
| [<img src="depth_estimation/fast-depth/output.png" width=256px>](depth_estimation/fast-depth/) |[fast-depth](depth_estimation/fast-depth)| [ICRA 2019 "FastDepth: Fast Monocular Depth Estimation on Embedded Systems"](https://github.com/dwofk/fast-depth) | Pytorch | 1.2.5 and later |
| [<img src="depth_estimation/lap-depth/out_kitti_demo.png" width=256px>](depth_estimation/lap-depth/) |[lap-depth](depth_estimation/lap-depth)| [LapDepth-release](https://github.com/tjqansthd/LapDepth-release) | Pytorch | 1.2.9 and later |
| [<img src="depth_estimation/hitnet/output.png" width=256px>](depth_estimation/hitnet/) |[hitnet](depth_estimation/hitnet)| [ONNX-HITNET-Stereo-Depth-estimation](https://github.com/ibaiGorordo/ONNX-HITNET-Stereo-Depth-estimation) | Pytorch | 1.2.9 and later |
 
## Face detection

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="face_detection/yolov1-face/output.png" width=128px>](face_detection/yolov1-face/) | [yolov1-face](/face_detection/yolov1-face/) | [YOLO-Face-detection](https://github.com/dannyblueliu/YOLO-Face-detection/) | Darknet | 1.1.0 and later |
| [<img src="face_detection/yolov3-face/output.png" width=128px>](face_detection/yolov3-face/) | [yolov3-face](/face_detection/yolov3-face/) | [Face detection using keras-yolov3](https://github.com/axinc-ai/yolov3-face) | Keras | 1.2.1 and later |
| [<img src="face_detection/blazeface/result.png" width=128px>](face_detection/blazeface/) |[blazeface](/face_detection/blazeface/)| [BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch) | Pytorch | 1.2.1 and later | [EN](https://medium.com/axinc-ai/blazeface-a-machine-learning-model-for-fast-detection-of-face-positions-and-key-points-5dcfb9429d72) [JP](https://medium.com/axinc/blazeface-%E9%A1%94%E3%81%AE%E4%BD%8D%E7%BD%AE%E3%81%A8%E3%82%AD%E3%83%BC%E3%83%9D%E3%82%A4%E3%83%B3%E3%83%88%E3%82%92%E9%AB%98%E9%80%9F%E3%81%AB%E6%A4%9C%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-e851c348a32b) |
| [<img src="face_detection/face-mask-detection/output.png" width=128px>](face_detection/face-mask-detection/)| [face-mask-detection](/face_detection/face-mask-detection/) | [Face detection using keras-yolov3](https://github.com/axinc-ai/yolov3-face) | Keras | 1.2.1 and later | [EN](https://medium.com/axinc-ai/facemaskdetection-a-machine-learning-model-to-determine-if-a-person-is-wearing-a-mask-e5a581ea8af9) [JP](https://medium.com/axinc/facemaskdetection-%E3%83%9E%E3%82%B9%E3%82%AF%E3%82%92%E4%BB%98%E3%81%91%E3%81%A6%E3%81%84%E3%82%8B%E3%81%8B%E3%82%92%E5%88%A4%E5%AE%9A%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-b06793f79a97) |
| [<img src="face_detection/dbface/selfie_output.png" width=128px>](face_detection/dbface/)| [dbface](face_detection/dbface/) | [DBFace : real-time, single-stage detector for face detection, <br/>with faster speed and higher accuracy](https://github.com/dlunion/DBFace) | Pytorch | 1.2.2 and later | 
| [<img src="face_detection/retinaface/selfie_output.png" width=128px>](face_detection/retinaface/)| [retinaface](face_detection/retinaface/) | [RetinaFace: Single-stage Dense Face Localisation in the Wild.](https://github.com/biubug6/Pytorch_Retinaface) | Pytorch | 1.2.5 and later | 
| [<img src="face_detection/anime-face-detector/output.png" width=128px>](face_detection/anime-face-detector/)| [anime-face-detector](face_detection/anime-face-detector/) | [Anime Face Detector](https://github.com/hysts/anime-face-detector) | Pytorch | 1.2.6 and later | 

## Face identification

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="face_identification/vggface2/couple_a.jpg">](face_identification/vggface2/) |[vggface2](/face_identification/vggface2) | [VGGFace2 Dataset for Face Recognition](https://github.com/ox-vgg/vgg_face2) | Caffe | 1.1.0 and later | |
| [<img src="face_identification/arcface/correct_pair_1.jpg" width=128px>](face_identification/arcface/)|[arcface](/face_identification/arcface) | [pytorch implement of arcface](https://github.com/ronghuaiyang/arcface-pytorch) | Pytorch | 1.2.1 and later | [EN](https://medium.com/axinc-ai/arcface-a-machine-learning-model-for-face-recognition-5f743cdac6fa) [JP](https://medium.com/axinc/arcface-%E9%A1%94%E8%AA%8D%E8%A8%BC%E3%82%92%E8%A1%8C%E3%81%86%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-cbb0e127bd0a) |
| [<img src="face_identification/insightface/output.png" width=128px>](face_identification/insightface/)|[insightface](/face_identification/insightface) | [InsightFace: 2D and 3D Face Analysis Project](https://github.com/deepinsight/insightface) | Pytorch | 1.2.5 and later | |

## Face recognition

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="face_recognition/face_classification/lenna.png">](face_recognition/face_classification/) |[face_classification](/face_recognition/face_classification) | [Real-time face detection and emotion/gender classification](https://github.com/oarriaga/face_classification) | Keras | 1.1.0 and later |
| [<img src="face_recognition/facial_feature/output.png" width=128px>](face_recognition/facial_feature/) |[facial_feature](/face_recognition/facial_feature/)|[kaggle-facial-keypoints](https://github.com/axinc-ai/kaggle-facial-keypoints)|Pytorch| 1.2.0 and later |
| [<img src="face_recognition/face_alignment/output_3d.png" width=128px>](face_recognition/face_alignment/) |[face_alignment](/face_recognition/face_alignment/)| [2D and 3D Face alignment library build using pytorch](https://github.com/1adrianb/face-alignment) | Pytorch | 1.2.1 and later | [EN](https://medium.com/axinc-ai/facealignment-a-machine-learning-model-for-recognizing-key-points-on-a-face-956f5e796efa) [JP](https://medium.com/axinc/facealignment-%E9%A1%94%E3%81%AE%E3%82%AD%E3%83%BC%E3%83%9D%E3%82%A4%E3%83%B3%E3%83%88%E3%82%92%E8%AA%8D%E8%AD%98%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-a46654c4da14) |
| [<img src="face_recognition/prnet/results/dense_alignment.png" width=128px>](face_recognition/prnet/) |[prnet](/face_recognition/prnet)| [Joint 3D Face Reconstruction and Dense Alignment <br/>with Position Map Regression Network](https://github.com/YadiraF/PRNet) | TensorFlow | 1.2.2 and later |
| [<img src="face_recognition/gazeml/output.png" width=128px>](face_recognition/gazeml/) | [gazeml](/face_recognition/gazeml/) | [A deep learning framework based on Tensorflow <br/>for the training of high performance gaze estimation](https://github.com/swook/GazeML) | TensorFlow | 1.2.0 and later |
| [<img src="face_recognition/facemesh/output.png" width=128px>](face_recognition/facemesh/) | [facemesh](/face_recognition/facemesh/) | [facemesh.pytorch](https://github.com/thepowerfuldeez/facemesh.pytorch) | Pytorch | 1.2.2 and later | [EN](https://medium.com/axinc-ai/facemesh-detecting-key-points-on-faces-in-real-time-977c03f1bab) [JP](https://medium.com/axinc/facemesh-%E3%83%AA%E3%82%A2%E3%83%AB%E3%82%BF%E3%82%A4%E3%83%A0%E3%81%A7%E9%A1%94%E3%81%AE%E3%82%AD%E3%83%BC%E3%83%9D%E3%82%A4%E3%83%B3%E3%83%88%E3%82%92%E6%A4%9C%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-bf223a50b7d6) |
| [<img src="face_recognition/mediapipe_iris/output.png" width=128px>](face_recognition/mediapipe_iris/) | [mediapipe_iris](/face_recognition/mediapipe_iris/) | [irislandmarks.pytorch](https://github.com/cedriclmenard/irislandmarks.pytorch) | Pytorch | 1.2.2 and later | [EN](https://medium.com/axinc-ai/mediapipe-iris-detecting-key-points-in-the-eye-637f5c1e728e) [JP](https://medium.com/axinc/mediapipe-iris-%E7%9B%AE%E3%81%AE%E3%82%AD%E3%83%BC%E3%83%9D%E3%82%A4%E3%83%B3%E3%83%88%E3%82%92%E6%A4%9C%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-a4742f143551) |
| [<img src="face_recognition/hopenet/output.png" width=128px>](face_recognition/hopenet/) | [hopenet](/face_recognition/hopenet/) | [deep-head-pose](https://github.com/natanielruiz/deep-head-pose) | Pytorch | 1.2.2 and later | [EN](https://medium.com/axinc-ai/hope-net-a-machine-learning-model-for-estimating-face-orientation-83d5af26a513) [JP](https://medium.com/axinc/hope-net-%E9%A1%94%E3%81%AE%E5%90%91%E3%81%8D%E3%82%92%E6%8E%A8%E5%AE%9A%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-6db21979f935) |
| [<img src="face_recognition/ax_gaze_estimation/output.png" width=128px>](face_recognition/ax_gaze_estimation/) | [ax_gaze_estimation](/face_recognition/ax_gaze_estimation/) | ax Gaze Estimation | Pytorch | 1.2.2 and later | [EN](https://medium.com/axinc-ai/axgazeestimation-a-machine-learning-model-for-estimating-gaze-c9648042d637) [JP](https://medium.com/axinc/axgazeestimation-%E8%A6%96%E7%B7%9A%E3%82%92%E6%8E%A8%E5%AE%9A%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-8446a791968) |
| [<img src="face_recognition/age-gender-recognition-retail/demo.jpg" width=128px>](face_recognition/age-gender-recognition-retail/) | [age-gender-recognition-retail](/face_recognition/age-gender-recognition-retail/) | [age-gender-recognition-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/age-gender-recognition-retail-0013) | OpenVINO | 1.2.5 and later | [EN](https://medium.com/axinc-ai/agegenderrecognitionretail-a-machine-learning-model-to-identify-age-and-gender-8506510414b) [JP](https://medium.com/axinc/agegenderrecognitionretail-%E5%B9%B4%E9%BD%A2%E3%81%A8%E6%80%A7%E5%88%A5%E3%82%92%E4%BA%88%E6%B8%AC%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-3632935d19ec) |
| [<img src="face_recognition/ferplus/img/fer0032227.png" width=128px>](face_recognition/ferplus/) | [ferplus](/face_recognition/ferplus/) | [FER+](https://github.com/microsoft/FERPlus) | CNTK | 1.2.2 and later |

## Frame Interpolation

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="frame_interpolation/flavr/sample_results/output.png" width=128px>](frame_interpolation/flavr/) | [flavr](/frame_interpolation/flavr/) | [FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation](https://github.com/tarun005/FLAVR) | Pytorch | 1.2.7 and later | [EN](https://medium.com/axinc-ai/flavr-a-machine-learning-model-to-increase-video-frame-rate-758fe8132818) [JP](https://medium.com/axinc/flavr-%E5%8B%95%E7%94%BB%E3%81%AE%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E3%83%AC%E3%83%BC%E3%83%88%E3%82%92%E4%B8%8A%E3%81%92%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-6a18211445da) |

## Generative adversarial networks

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="generative_adversarial_networks/pytorch-gan/output_anime.png">](generative_adversarial_networks/pytorch-gan/) |[pytorch-gan](/generative_adversarial_networks/pytorch-gan) | [Code repo for the Pytorch GAN Zoo project (used to train this model)](https://github.com/facebookresearch/pytorch_GAN_zoo)| Pytorch | 1.2.4 and later |
| [<img src="generative_adversarial_networks/council-GAN/output_glasses.png" width=128px>](generative_adversarial_networks/council-GAN/) | [council-GAN](/generative_adversarial_networks/council-GAN)| [Council-GAN](https://github.com/Onr/Council-GAN)| Pytorch | 1.2.4 and later |
| [<img src="generative_adversarial_networks/restyle-encoder/img/face_img.jpg" width=128px>](generative_adversarial_networks/restyle-encoder/) | [restyle-encoder](/generative_adversarial_networks/restyle-encoder)| [ReStyle](https://github.com/yuval-alaluf/restyle-encoder)| Pytorch | 1.2.9 and later |
| [<img src="generative_adversarial_networks/sam/img/input.jpg" width=128px>](generative_adversarial_networks/sam/) | [sam](/generative_adversarial_networks/sam)| [Age Transformation Using a Style-Based Regression Model](https://github.com/yuval-alaluf/SAM)| Pytorch | 1.2.9 and later |

## Hand detection

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="hand_detection/yolov3-hand/output.png" width=128px>](hand_detection/yolov3-hand/) | [yolov3-hand](/hand_detection/yolov3-hand/) | [Hand detection branch of Face detection using keras-yolov3](https://github.com/axinc-ai/yolov3-face/tree/hand_detection) | Keras | 1.2.1 and later |
| [<img src="hand_detection/hand_detection_pytorch/CARDS_OFFICE_output.jpg" width=128px>](hand_detection/hand_detection_pytorch/) | [hand_detection_pytorch](/hand_detection/hand_detection_pytorch/) | [hand-detection.PyTorch](https://github.com/zllrunning/hand-detection.PyTorch) | Pytorch | 1.2.2 and later |
| [<img src="hand_detection/blazepalm/output.png" width=128px>](hand_detection/blazepalm/) |[blazepalm](/hand_detection/blazepalm/) | [MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch) | Pytorch | 1.2.5 and later |

## Hand recognition

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="hand_recognition/blazehand/output.png" width=128px>](hand_recognition/blazehand/) |[blazehand](/hand_recognition/blazehand/) | [MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch) | Pytorch | 1.2.5 and later | [EN](https://medium.com/axinc-ai/blazehand-a-machine-learning-model-for-detecting-hand-key-points-c3943b82739a) [JP](https://medium.com/axinc/blazehand-%E6%89%8B%E3%81%AE%E3%82%AD%E3%83%BC%E3%83%9D%E3%82%A4%E3%83%B3%E3%83%88%E3%82%92%E6%A4%9C%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-e84e011ef7bc) |
| [<img src="hand_recognition/hand3d/output.png" width=128px>](hand_recognition/hand3d/) |[hand3d](/hand_recognition/hand3d/) | [ColorHandPose3D network](https://github.com/lmb-freiburg/hand3d) | TensorFlow | 1.2.5 and later |
| [<img src="hand_recognition/minimal-hand/output.png" width=128px>](hand_recognition/minimal-hand/) |[minimal-hand](/hand_recognition/minimal-hand/) | [Minimal Hand](https://github.com/CalciferZh/minimal-hand) | TensorFlow | 1.2.8 and later |

## Image captioning

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="image_captioning/illustration2vec/input.jpg" width=128px>](image_captioning/illustration2vec/) | [illustration2vec](/image_captioning/illustration2vec/)|[Illustration2Vec](https://github.com/rezoo/illustration2vec) | Caffe | 1.2.2 and later |
| [<img src="image_captioning/image_captioning_pytorch/demo.jpg" width=128px>](image_captioning/image_captioning_pytorch/) | [image_captioning_pytorch](/image_captioning/image_captioning_pytorch/)|[Image Captioning pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch) | Pytorch | 1.2.5 and later | [EN](https://medium.com/axinc-ai/image-captioning-pytorch-a-machine-learning-model-for-describing-images-d27562b6f15b) [JP](https://medium.com/axinc/image-captioning-pytorch-%E7%94%BB%E5%83%8F%E3%82%92%E8%AA%AC%E6%98%8E%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-e690982af19) |

## Image classification

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="image_classification/vgg16/clock.jpg" width=128px>](image_classification/vgg16/) | [vgg16](/image_classification/vgg16/) |[Very Deep Convolutional Networks for Large-Scale Image Recognition]( https://arxiv.org/abs/1409.1556 )|Keras| 1.1.0 and later| |
| [<img src="image_classification/googlenet/pizza.jpg" width=128px>](image_classification/googlenet/) | [googlenet](/image_classification/googlenet/) |[Going Deeper with Convolutions]( https://arxiv.org/abs/1409.4842 )|Pytorch| 1.2.0 and later| |
| [<img src="image_classification/resnet50/pizza.jpg" width=128px>](image_classification/resnet50/) | [resnet50](/image_classification/resnet50/) | [Deep Residual Learning for Image Recognition]( https://github.com/KaimingHe/deep-residual-networks) | Chainer | 1.2.0 and later | |
| [<img src="image_classification/inceptionv3/clock.jpg" width=128px>](image_classification/inceptionv3/) | [inceptionv3](/image_classification/inceptionv3/)|[Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)|Pytorch| 1.2.0 and later | [JP](https://medium.com/axinc/ailia-sdk-%E3%83%A2%E3%83%87%E3%83%AB%E7%B4%B9%E4%BB%8B-inceptionv3-b39dd43f285d) |
| [<img src="image_classification/inceptionv4/clock.jpg" width=128px>](image_classification/inceptionv4/) | [inceptionv4](/image_classification/inceptionv4/)|[Keras Inception-V4](https://github.com/kentsommer/keras-inceptionV4)|Keras| 1.2.5 and later | |
| [<img src="image_classification/mobilenetv2/clock.jpg" width=128px>](image_classification/mobilenetv2/) | [mobilenetv2](/image_classification/mobilenetv2/)|[PyTorch Implemention of MobileNet V2](https://github.com/d-li14/mobilenetv2.pytorch)|Pytorch| 1.2.0 and later | |
| [<img src="image_classification/mobilenetv3/clock.jpg" width=128px>](image_classification/mobilenetv3/) | [mobilenetv3](/image_classification/mobilenetv3/)|[PyTorch Implemention of MobileNet V3](https://github.com/d-li14/mobilenetv3.pytorch)|Pytorch| 1.2.1 and later | |
| [<img src="image_classification/partialconv/test_5735.JPEG" width=128px>](image_classification/partialconv/) | [partialconv](/image_classification/partialconv/)|[Partial Convolution Layer for Padding and Image Inpainting](https://github.com/NVIDIA/partialconv)|Pytorch| 1.2.0 and later | |
| [<img src="image_classification/efficientnet/clock.jpg" width=128px>](image_classification/efficientnet/)| [efficientnet](/image_classification/efficientnet/)|[A PyTorch implementation of EfficientNet]( https://github.com/lukemelas/EfficientNet-PyTorch)|Pytorch| 1.2.3 and later | |
| [<img src="image_classification/efficientnetv2/input.jpg" width=128px>](image_classification/efficientnetv2/)| [efficientnetv2](/image_classification/efficientnetv2/)|[EfficientNetV2]( https://github.com/google/automl/tree/master/efficientnetv2 )|Pytorch| 1.2.4 and later | |
| [<img src="image_classification/vit/output.png" width=128px>](image_classification/vit/)| [vit](/image_classification/vit/)|[Pytorch reimplementation of the Vision Transformer (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)](https://github.com/jeonsworld/ViT-pytorch)|Pytorch| 1.2.7 and later | [JP](https://medium.com/axinc/vision-transformer-%E7%95%B3%E3%81%BF%E8%BE%BC%E3%81%BF%E6%BC%94%E7%AE%97%E3%82%92%E7%94%A8%E3%81%84%E3%81%AA%E3%81%84%E6%9C%80%E6%96%B0%E7%94%BB%E5%83%8F%E8%AD%98%E5%88%A5%E6%8A%80%E8%A1%93-84f06978a17f) |
| [<img src="image_classification/wide_resnet50/dog.jpg" width=128px>](image_classification/wide_resnet50/)| [wide_resnet50](/image_classification/wide_resnet50/)|[Wide Resnet](https://pytorch.org/hub/pytorch_vision_wide_resnet/)|Pytorch| 1.2.5 and later | |
| [<img src="image_classification/mlp_mixer/input.jpg" width=128px>](image_classification/mlp_mixer/)| [mlp_mixer](/image_classification/mlp_mixer/)|[MLP-Mixer](https://github.com/jeonsworld/MLP-Mixer-Pytorch)|Pytorch| 1.2.9 and later | |

## Image inpainting

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="image_inpainting/pytorch-inpainting-with-partial-conv/result.png" width=128px>](image_inpainting/pytorch-inpainting-with-partial-conv/) | [inpainting-with-partial-conv](/image_inpainting/pytorch-inpainting-with-partial-conv/) | [pytorch-inpainting-with-partial-conv](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv) | PyTorch | 1.2.6 and later | [EN](https://medium.com/axinc-ai/inpainting-with-partial-conv-a-machine-learning-model-that-predicts-and-fills-in-missing-parts-of-53c046343a85) [JP](https://medium.com/axinc/inpainting-with-partial-conv-%E7%94%BB%E5%83%8F%E3%81%AE%E6%AC%A0%E6%90%8D%E9%83%A8%E5%88%86%E3%82%92%E4%BA%88%E6%B8%AC%E3%81%97%E3%81%A6%E5%9F%8B%E3%82%81%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-9746576e6490) |
| [<img src="image_inpainting/inpainting_gmcnn/result_paris-streetview_rect.png" width=128px>](image_inpainting/inpainting_gmcnn/) | [inpainting_gmcnn](/image_inpainting/inpainting_gmcnn/) | [Image Inpainting via Generative Multi-column Convolutional Neural Networks](https://github.com/shepnerd/inpainting_gmcnn) | TensorFlow | 1.2.6 and later |
| [<img src="image_inpainting/3d-photo-inpainting/example_moon_circle.jpg" width=128px>](image_inpainting/3d-photo-inpainting/) | [3d-photo-inpainting](/image_inpainting/3d-photo-inpainting/) | [3D Photography using Context-aware Layered Depth Inpainting](https://github.com/vt-vl-lab/3d-photo-inpainting) | Pytorch | 1.2.7 and later |

## Image manipulation

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="image_manipulation/noise2noise/output.png" width=128px>](image_manipulation/noise2noise/) | [noise2noise](/image_manipulation/noise2noise/) | [Learning Image Restoration without Clean Data](https://github.com/joeylitalien/noise2noise-pytorch) | Pytorch | 1.2.0 and later |
| [<img src="image_manipulation/dewarpnet/output.png" width=128px>](image_manipulation/dewarpnet/) | [dewarpnet](/image_manipulation/dewarpnet) | [DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks](https://github.com/cvlab-stonybrook/DewarpNet) | Pytorch | 1.2.1 and later |
| [<img src="image_manipulation/illnet/output.png" width=128px>](image_manipulation/illnet/) | [illnet](/image_manipulation/illnet/) | [Document Rectification and Illumination Correction using a Patch-based CNN](https://github.com/xiaoyu258/DocProj) | Pytorch | 1.2.2 and later |
| [<img src="image_manipulation/colorization/imgs_out/ansel_adams3_output.jpg" width=128px>](image_manipulation/colorization/) | [colorization](/image_manipulation/colorization/) | [Colorful Image Colorization](https://github.com/richzhang/colorization) | Pytorch | 1.2.2 and later | [EN](https://medium.com/axinc-ai/colorization-a-machine-learning-model-for-colorizing-black-and-white-images-829e35e4f91c) [JP](https://medium.com/axinc/colorization-%E7%99%BD%E9%BB%92%E7%94%BB%E5%83%8F%E3%82%92%E3%82%AB%E3%83%A9%E3%83%BC%E5%8C%96%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-177d3fd52e40) |
| [<img src="image_manipulation/u2net_portrait/your_portrait_results/GalGadot.jpg" width=128px>](image_manipulation/u2net_portrait/) | [u2net_portrait](/image_manipulation/u2net_portrait/) | [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/NathanUA/U-2-Net) | Pytorch | 1.2.2 and later |
| [<img src="image_manipulation/style2paints/output.png" width=128px>](image_manipulation/style2paints/) | [style2paints](/image_manipulation/style2paints/) | [Style2Paints](https://github.com/lllyasviel/style2paints) | TensorFlow | 1.2.6 and later |
| [<img src="image_manipulation/deep_white_balance/output_AWB.png" width=128px>](image_manipulation/deep_white_balance/) | [deep_white_balance](/image_manipulation/deep_white_balance/) | [Deep White-Balance Editing, CVPR 2020 (Oral)](https://github.com/mahmoudnafifi/Deep_White_Balance) | PyTorch | 1.2.6 and later |
| [<img src="image_manipulation/deblur_gan/output.png" width=128px>](image_manipulation/deblur_gan/) | [deblur_gan](/image_manipulation/deblur_gan/) | [DeblurGAN](https://github.com/KupynOrest/DeblurGAN) | Pytorch | 1.2.6 and later |
| [<img src="image_manipulation/dpr/output.png" width=128px>](image_manipulation/dpr/) | [dpr](/image_manipulation/dpr/) | [Deep Single-Image Portrait Relighting](https://github.com/zhhoper/DPR) | Pytorch | 1.2.8 and later |

## Image segmentation

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="image_segmentation/deeplabv3/output.png" width=128px>](image_segmentation/deeplabv3/) | [deeplabv3](/image_segmentation/deeplabv3/) | [Xception65 for backbone network of DeepLab v3+](https://github.com/tensorflow/models/tree/master/research/deeplab) | Chainer | 1.2.0 and later |
| [<img src="image_segmentation/hrnet_segmentation/result.png" width=128px>](image_segmentation/hrnet_segmentation/) | [hrnet_segmentation](/image_segmentation/hrnet_segmentation/) | [High-resolution networks (HRNets) for Semantic Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) | Pytorch | 1.2.1 and later | |
| [<img src="image_segmentation/hair_segmentation/output.png" width=128px>](image_segmentation/hair_segmentation/) | [hair_segmentation](/image_segmentation/hair_segmentation/) | [hair segmentation in mobile device](https://github.com/thangtran480/hair-segmentation) | Keras | 1.2.1 and later | |
| [<img src="image_segmentation/pspnet-hair-segmentation/output.png" width=128px>](image_segmentation/pspnet-hair-segmentation/) | [pspnet-hair-segmentation](/image_segmentation/pspnet-hair-segmentation/) | [pytorch-hair-segmentation](https://github.com/YBIGTA/pytorch-hair-segmentation) | Pytorch | 1.2.2 and later | |
| [<img src="image_segmentation/human_part_segmentation/output.png" width=128px>](image_segmentation/human_part_segmentation/) | [human_part_segmentation](/image_segmentation/human_part_segmentation/) | [Self Correction for Human Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing) | Pytorch | 1.2.4 and later | [EN](https://medium.com/axinc-ai/humanpartsegmentation-a-machine-learning-model-for-segmenting-human-parts-cd7e39480714) [JP](https://medium.com/axinc/humanpartsegmentation-%E5%8B%95%E7%94%BB%E3%81%8B%E3%82%89%E4%BD%93%E3%81%AE%E9%83%A8%E4%BD%8D%E3%82%92%E3%82%BB%E3%82%B0%E3%83%A1%E3%83%B3%E3%83%86%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-e8a0e405255) |
| [<img src="image_segmentation/semantic-segmentation-mobilenet-v3/output.png" width=128px>](image_segmentation/semantic-segmentation-mobilenet-v3/) | [semantic-segmentation-mobilenet-v3](/image_segmentation/semantic-segmentation-mobilenet-v3) | [Semantic segmentation with MobileNetV3](https://github.com/OniroAI/Semantic-segmentation-with-MobileNetV3) | TensorFlow | 1.2.5 and later | |
| [<img src="image_segmentation/pytorch-unet/data/masks/0cdf5b5d0ce1_01.jpg" width=128px>](image_segmentation/pytorch-unet/) | [pytorch-unet](/image_segmentation/pytorch-unet/) | [Pytorch-Unet](https://github.com/milesial/Pytorch-UNet) | Pytorch | 1.2.5 and later | |
| [<img src="image_segmentation/pytorch-enet/output.png" width=128px>](image_segmentation/pytorch-enet/) | [pytorch-enet](/image_segmentation/pytorch-enet/) | [PyTorch-ENet](https://github.com/davidtvs/PyTorch-ENet) | Pytorch | 1.2.8 and later | |
| [<img src="image_segmentation/yet-another-anime-segmenter/output.png" width=128px>](image_segmentation/yet-another-anime-segmenter/) | [yet-another-anime-segmenter](/image_segmentation/yet-another-anime-segmenter/) | [Yet Another Anime Segmenter](https://github.com/zymk9/Yet-Another-Anime-Segmenter) | Pytorch | 1.2.6 and later | |
| [<img src="image_segmentation/swiftnet/output.png" width=128px>](image_segmentation/swiftnet/) | [swiftnet](/image_segmentation/swiftnet/) | [SwiftNet](https://github.com/orsic/swiftnet) | Pytorch | 1.2.6 and later | |
| [<img src="image_segmentation/dense_prediction_transformers/output_segmentation.png" width=128px>](image_segmentation/dense_prediction_transformers/) | [dense_prediction_transformers](/image_segmentation/dense_prediction_transformers/) | [Vision Transformers for Dense Prediction](https://github.com/intel-isl/DPT)   | Pytorch | 1.2.7 and later | [EN](https://medium.com/axinc-ai/dpt-segmentation-model-using-vision-transformer-b479f3027468) [JP](https://medium.com/axinc/dpt-vision-transformer%E3%82%92%E4%BD%BF%E7%94%A8%E3%81%97%E3%81%9F%E3%82%BB%E3%82%B0%E3%83%A1%E3%83%B3%E3%83%86%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%83%A2%E3%83%87%E3%83%AB-88db4842b4a7) |

## Line segment detection

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="line_segment_detection/mlsd/output.jpg" width=128px>](line_segment_detection/mlsd/) | [mlsd](/line_segment_detection/mlsd/) | [M-LSD: Towards Light-weight and Real-time Line Segment Detection](https://github.com/navervision/mlsd) | TensorFlow | 1.2.8 and later |  [EN](https://medium.com/axinc-ai/m-lsd-machine-learning-model-for-detecting-wireframes-ac1b618f459b) [JP](https://medium.com/axinc/m-lsd-%E3%83%AF%E3%82%A4%E3%83%A4%E3%83%BC%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E3%82%92%E6%A4%9C%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-876bef18d908) |

## Natural language processing

| Model | Reference | Exported From | Supported Ailia Version | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|
|[bert](/neural_language_processing/bert) | [pytorch-pretrained-bert](https://pypi.org/project/pytorch-pretrained-bert/) | Pytorch | 1.2.2 and later | [EN](https://medium.com/axinc-ai/bert-a-machine-learning-model-for-efficient-natural-language-processing-aef3081c24e8) [JP](https://medium.com/axinc/bert-%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%82%92%E5%8A%B9%E7%8E%87%E7%9A%84%E3%81%AB%E5%AD%A6%E7%BF%92%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-3a9c27d78cf8) |
|[bert_maskedlm](/neural_language_processing/bert_maskedlm) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later |
|[bert_ner](/neural_language_processing/bert_ner) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later |
|[bert_question_answering](/neural_language_processing/bert_question_answering) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later |
|[bert_sentiment_analysis](/neural_language_processing/bert_sentiment_analysis) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later |
|[bert_zero_shot_classification](/neural_language_processing/bert_zero_shot_classification) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later |
|[bert_tweets_sentiment](/neural_language_processing/bert_tweets_sentiment) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later |
|[gpt2](/neural_language_processing/gpt2) | [GPT-2](https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/README.md) | Pytorch | 1.2.7 and later |
|[rinna_gpt2](/neural_language_processing/rinna_gpt2) | [japanese-pretrained-models](https://github.com/rinnakk/japanese-pretrained-models)   | Pytorch | 1.2.7 and later |

## Object detection

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="object_detection/yolov1-tiny/output.png" width=128px>](object_detection/yolov1-tiny/) | [yolov1-tiny](/object_detection/yolov1-tiny/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolov1/) | Darknet | 1.1.0 and later |
| [<img src="object_detection/yolov2/output.png" width=128px>](object_detection/yolov2/) | [yolov2](/object_detection/yolov2/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | Pytorch | 1.2.0 and later |
| [<img src="object_detection/yolov3/output.png" width=128px>](object_detection/yolov3/) | [yolov3](/object_detection/yolov3/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | ONNX Runtime | 1.2.1 and later | [EN](https://medium.com/axinc-ai/yolov3-a-machine-learning-model-to-detect-the-position-and-type-of-an-object-60f1c18f8107) [JP](https://medium.com/axinc/yolov3-66c9b998c096) |
| [<img src="object_detection/yolov3-tiny/output.png" width=128px>](object_detection/yolov3-tiny/) | [yolov3-tiny](/object_detection/yolov3-tiny/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | ONNX Runtime | 1.2.1 and later |
| [<img src="object_detection/yolov4/output.png" width=128px>](object_detection/yolov4/) | [yolov4](/object_detection/yolov4/) | [Pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) | Pytorch | 1.2.4 and later | [EN](https://medium.com/axinc-ai/yolov4-a-machine-learning-model-to-detect-the-position-and-type-of-an-object-4f108ed0507b) [JP](https://medium.com/axinc/yolov4-%E7%89%A9%E4%BD%93%E3%82%92%E6%A4%9C%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-480f0a635317) |
| [<img src="object_detection/yolov4-tiny/output.png" width=128px>](object_detection/yolov4-tiny/) | [yolov4-tiny](/object_detection/yolov4-tiny/) | [Pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) | Pytorch | 1.2.5 and later |
| [<img src="object_detection/yolov5/output.png" width=128px>](object_detection/yolov5/) | [yolov5](/object_detection/yolov5/) | [yolov5](https://github.com/ultralytics/yolov5) | Pytorch | 1.2.5 and later | [EN](https://medium.com/axinc-ai/yolov5-the-latest-model-for-object-detection-b13320ec516b) [JP](https://medium.com/axinc/yolov5-%E7%89%A9%E4%BD%93%E6%A4%9C%E5%87%BA%E3%81%AE%E6%9C%80%E6%96%B0%E3%83%A2%E3%83%87%E3%83%AB-5b7316d1e54d) |
| [<img src="object_detection/yolor/output.jpg" width=128px>](object_detection/yolor/) | [yolor](/object_detection/yolor/) | [yolor](https://github.com/WongKinYiu/yolor/tree/paper) | Pytorch | 1.2.5 and later |
| [<img src="object_detection/yolox/output.jpg" width=128px>](object_detection/yolox/) | [yolox](/object_detection/yolox/) | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Pytorch | 1.2.6 and later | [EN](https://medium.com/axinc-ai/yolox-object-detection-model-exceeding-yolov5-d6cea6d3c4bc) [JP](https://medium.com/axinc/yolox-yolov5%E3%82%92%E8%B6%85%E3%81%88%E3%82%8B%E7%89%A9%E4%BD%93%E6%A4%9C%E5%87%BA%E3%83%A2%E3%83%87%E3%83%AB-e9706e15fef2) |
| [<img src="object_detection/mobilenet_ssd/output.png" width=128px>](object_detection/mobilenet_ssd/) | [mobilenet_ssd](/object_detection/mobilenet_ssd/) | [MobileNetV1, MobileNetV2, VGG based SSD/SSD-lite implementation in Pytorch](https://github.com/qfgaohao/pytorch-ssd) | Pytorch | 1.2.1 and later | [EN](https://medium.com/axinc-ai/mobilenetssd-a-machine-learning-model-for-fast-object-detection-37352ce6da7d) [JP](https://medium.com/axinc/mobilenetssd-%E9%AB%98%E9%80%9F%E3%81%AB%E7%89%A9%E4%BD%93%E6%A4%9C%E5%87%BA%E3%82%92%E8%A1%8C%E3%81%86%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-be3ca37c411) |
| [<img src="object_detection/maskrcnn/output.png" width=128px>](object_detection/maskrcnn/) | [maskrcnn](/object_detection/maskrcnn/) | [Mask R-CNN: real-time neural network for object instance segmentation](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/mask-rcnn) | Pytorch | 1.2.3 and later |
| [<img src="object_detection/m2det/output.png" width=128px>](object_detection/m2det/) | [m2det](/object_detection/m2det/) | [M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://github.com/qijiezhao/M2Det) | Pytorch | 1.2.3 and later | [EN](https://medium.com/axinc-ai/m2det-highly-accurate-object-detection-model-b5c5bff27970) [JP](https://medium.com/axinc/m2det-%E9%AB%98%E7%B2%BE%E5%BA%A6%E3%81%AA%E7%89%A9%E4%BD%93%E6%A4%9C%E5%87%BA%E3%83%A2%E3%83%87%E3%83%AB-bf92a8a3d423) |
| [<img src="object_detection/centernet/output.png" width=128px>](object_detection/centernet/) | [centernet](/object_detection/centernet/) | [CenterNet : Objects as Points](https://github.com/xingyizhou/CenterNet) | Pytorch | 1.2.1 and later | [EN](https://medium.com/axinc-ai/centernet-a-machine-learning-model-for-anchorless-object-detection-462c48483cfe) [JP](https://medium.com/axinc/centernet-%E3%82%A2%E3%83%B3%E3%82%AB%E3%83%BC%E3%83%AC%E3%82%B9%E3%81%AA%E7%89%A9%E4%BD%93%E6%A4%9C%E5%87%BA%E3%82%92%E8%A1%8C%E3%81%86%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-9ecbadefd884) |
| [<img src="object_detection/pedestrian_detection/output.png" width=128px>](object_detection/pedestrian_detection/) | [pedestrian_detection](/object_detection/pedestrian_detection/) | [Pedestrian-Detection-on-YOLOv3_Research-and-APP](https://github.com/Zyjacya-In-love/Pedestrian-Detection-on-YOLOv3_Research-and-APP) | Keras | 1.2.1 and later |
| [<img src="object_detection/efficientdet/output.png" width=128px>](object_detection/efficientdet/) | [efficientdet](/object_detection/efficientdet/) | [EfficientDet: Scalable and Efficient Object Detection, in PyTorch](https://github.com/toandaominh1997/EfficientDet.Pytorch) | Pytorch | 1.2.6 and later |
| [<img src="object_detection/nanodet/output.jpg" width=128px>](object_detection/nanodet/) | [nanodet](/object_detection/nanodet/) | [NanoDet](https://github.com/RangiLyu/nanodet) | Pytorch | 1.2.6 and later |

## Object detection 3d

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="object_detection_3d/3d_bbox/output.png" width=128px>](object_detection_3d/efficientdet/) | [3d_bbox](/object_detection_3d/3d_bbox/) | [3D Bounding Box Estimation Using Deep Learning and Geometry](https://github.com/skhadem/3D-BoundingBox) | Pytorch | 1.2.6 and later | |
| [<img src="object_detection_3d/3d-object-detection.pytorch/output.png" width=128px>](object_detection_3d/3d-object-detection.pytorch/) | [3d-object-detection.pytorch](/object_detection_3d/3d-object-detection.pytorch/) | [3d-object-detection.pytorch](https://github.com/sovrasov/3d-object-detection.pytorch) | Pytorch | 1.2.8 and later | [EN](https://medium.com/axinc-ai/3dobjectdetectionpytorch-3d-object-detection-model-3e1da4c53f) [JP](https://medium.com/axinc/3dobjectdetectionpytorch-3d%E3%81%AE%E7%89%A9%E4%BD%93%E6%A4%9C%E5%87%BA%E3%83%A2%E3%83%87%E3%83%AB-8df18b8eb5d1) |
| [<img src="object_detection_3d/mediapipe_objectron/output.png" width=128px>](object_detection_3d/mediapipe_objectron/) | [mediapipe_objectron](/object_detection_3d/mediapipe_objectron/) | [MediaPipe Objectron](https://github.com/google/mediapipe) | TensorFlow Lite | 1.2.5 and later | |

## Object tracking

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="object_tracking/deepsort/demo.gif" width=128px>](object_tracking/deepsort/) | [deepsort](/object_tracking/deepsort/) | [Deep Sort with PyTorch](https://github.com/ZQPei/deep_sort_pytorch) | Pytorch | 1.2.3 and later | [EN](https://medium.com/axinc-ai/deepsort-a-machine-learning-model-for-tracking-people-1170743b5984) [JP](https://medium.com/axinc/deepsort-%E4%BA%BA%E7%89%A9%E3%81%AE%E3%83%88%E3%83%A9%E3%83%83%E3%82%AD%E3%83%B3%E3%82%B0%E3%82%92%E8%A1%8C%E3%81%86%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-e8cb7410457c) |
| [<img src="object_tracking/person_reid_baseline_pytorch/output.png" width=128px>](object_tracking/person_reid_baseline_pytorch/) | [person_reid_baseline_pytorch](/object_tracking/person_reid_baseline_pytorch/) | [UTS-Person-reID-Practical](https://github.com/layumi/Person_reID_baseline_pytorch) | Pytorch | 1.2.6 and later | |
| [<img src="object_tracking/abd_net/output.png" width=128px>](object_tracking/abd_net/) | [abd_net](/object_tracking/abd_net/) | [Attentive but Diverse Person Re-Identification](https://github.com/VITA-Group/ABD-Net) | Pytorch | 1.2.7 and later | |
| [<img src="object_tracking/siam-mot/output.png" width=128px>](object_tracking/siam-mot/) | [siam-mot](/object_tracking/siam-mot/) | [SiamMOT](https://github.com/amazon-research/siam-mot) | Pytorch | 1.2.9 and later | |
| [<img src="object_tracking/bytetrack/output.png" width=128px>](object_tracking/bytetrack/) | [bytetrack](/object_tracking/bytetrack/) | [ByteTrack](https://github.com/ifzhang/ByteTrack) | Pytorch | 1.2.5 and later | [JP](https://medium.com/axinc/bytetrack-%E4%BD%8E%E3%81%84%E7%A2%BA%E5%BA%A6%E3%81%AEboundingbox%E3%82%82%E8%80%83%E6%85%AE%E3%81%99%E3%82%8B%E3%83%88%E3%83%A9%E3%83%83%E3%82%AD%E3%83%B3%E3%82%B0%E3%83%A2%E3%83%87%E3%83%AB-244b994d5afb)　|

## Optical Flow Estimation

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="optical_flow_estimation/raft/output.png" width=128px>](optical_flow_estimation/raft/) | [raft](/optical_flow_estimation/raft/) | [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://github.com/princeton-vl/RAFT) | Pytorch | 1.2.6 and later | [JP](https://medium.com/axinc/raft-optical-flow%E3%82%92%E6%8E%A8%E5%AE%9A%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-bf898965de05)　|

## Point segmentation

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="point_segmentation/pointnet_pytorch/output.png" width=128px>](point_segmentation/pointnet_pytorch/) | [pointnet_pytorch](/point_segmentation/pointnet_pytorch/) | [PointNet.pytorch](https://github.com/fxia22/pointnet.pytorch) | Pytorch | 1.2.6 and later |

## Pose estimation

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="pose_estimation/openpose/output.png" width=128px>](pose_estimation/openpose/) |[openpose](/pose_estimation/openpose/) | [Code repo for realtime multi-person pose estimation in CVPR'17 (Oral)](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) | Caffe | 1.2.1 and later |
| [<img src="pose_estimation/lightweight-human-pose-estimation/output.png" width=128px>](pose_estimation/lightweight-human-pose-estimation/)  |[lightweight-human-pose-estimation](/pose_estimation/lightweight-human-pose-estimation/) | [Fast and accurate human pose estimation in PyTorch.<br/>Contains implementation of <br/>"Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose" paper.](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) | Pytorch | 1.2.1 and later | [EN](https://medium.com/axinc-ai/lightweighthumanpose-a-machine-learning-model-for-fast-multi-person-skeleton-detection-631c042bed50) [JP](https://medium.com/axinc/lightweighthumanpose-%E9%AB%98%E9%80%9F%E3%81%AB%E8%A4%87%E6%95%B0%E4%BA%BA%E3%81%AE%E9%AA%A8%E6%A0%BC%E3%82%92%E6%A4%9C%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-bc34d420e6e2) |
| [<img src="pose_estimation/pose_resnet/output.png" width=128px>](pose_estimation/pose_resnet/) |[pose_resnet](/pose_estimation/pose_resnet/) | [Simple Baselines for Human Pose Estimation and Tracking](https://github.com/microsoft/human-pose-estimation.pytorch) | Pytorch | 1.2.1 and later | [EN](https://medium.com/axinc-ai/poseresnet-a-top-down-machine-learning-model-for-skeletal-detection-9454f391ae4d) [JP](https://medium.com/axinc/poseresnet-%E3%83%88%E3%83%83%E3%83%97%E3%83%80%E3%82%A6%E3%83%B3%E3%81%A7%E9%AA%A8%E6%A0%BC%E6%A4%9C%E5%87%BA%E3%82%92%E8%A1%8C%E3%81%86%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-9e0d20396d1e) |
| [<img src="pose_estimation/blazepose/output.png" width=128px>](pose_estimation/blazepose/) |[blazepose](/pose_estimation/blazepose/) | [MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch) | Pytorch | 1.2.5 and later |
| [<img src="pose_estimation/efficientpose/output.png" width=128px>](pose_estimation/efficientpose/) |[efficientpose](/pose_estimation/efficientpose/) | [Code repo for EfficientPose](https://github.com/daniegr/EfficientPose) | TensorFlow | 1.2.6 and later |
| [<img src="pose_estimation/movenet/output.png" width=128px>](pose_estimation/movenet/) |[movenet](/pose_estimation/movenet/) | [Code repo for movenet](https://www.tensorflow.org/hub/tutorials/movenet) | TensorFlow | 1.2.8 and later | [EN](https://medium.com/axinc-ai/movenet-pose-estimation-for-video-with-intense-motion-2b92f53f3c8) [JP](https://medium.com/axinc/movenet-%E5%8B%95%E3%81%8D%E3%81%AE%E6%BF%80%E3%81%97%E3%81%84%E5%8B%95%E7%94%BB%E5%90%91%E3%81%91%E9%AA%A8%E6%A0%BC%E6%A4%9C%E5%87%BA%E3%83%A2%E3%83%87%E3%83%AB-d26d9e06126c)|
| [<img src="pose_estimation/animalpose/output.png" width=128px>](pose_estimation/animalpose/) |[animalpose](/pose_estimation/animalpose/) | [MMPose - 2D animal pose estimation](https://github.com/open-mmlab/mmpose) | Pytorch | 1.2.7 and later | [EN](https://medium.com/axinc-ai/animalpose-pose-esimation-for-animals-700603e0dbae) [JP](https://medium.com/axinc/animalpose-%E5%8B%95%E7%89%A9%E3%81%AE%E9%AA%A8%E6%A0%BC%E6%A4%9C%E5%87%BA%E3%83%A2%E3%83%87%E3%83%AB-f7f667c0e69d) |

## Pose estimation 3d

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="pose_estimation_3d/lightweight-human-pose-estimation-3d/output.png" width=128px>](pose_estimation_3d/lightweight-human-pose-estimation-3d/) |[lightweight-human-pose-estimation-3d](/pose_estimation_3d/lightweight-human-pose-estimation-3d/) | [Real-time 3D multi-person pose estimation demo in PyTorch.<br/>OpenVINO backend can be used for fast inference on CPU.](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch) | Pytorch | 1.2.1 and later | |
| [<img src="pose_estimation_3d/3d-pose-baseline/output.png" width=128px>](pose_estimation_3d/3d-pose-baseline/) |[3d-pose-baseline](/pose_estimation_3d/3d-pose-baseline/) | [A simple baseline for 3d human pose estimation in tensorflow.<br/>Presented at ICCV 17.](https://github.com/una-dinosauria/3d-pose-baseline) | TensorFlow | 1.2.3 and later | |
| [<img src="pose_estimation_3d/pose-hg-3d/output.png" width=128px>](pose_estimation_3d/pose-hg-3d/) |[pose-hg-3d](/pose_estimation_3d/pose-hg-3d/) | [Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach](https://github.com/xingyizhou/pytorch-pose-hg-3d) | Pytorch | 1.2.6 and later | |
| [<img src="pose_estimation_3d/blazepose-fullbody/output.png" width=128px>](pose_estimation_3d/blazepose-fullbody/) |[blazepose-fullbody](/pose_estimation_3d/blazepose-fullbody/) | [MediaPipe](https://google.github.io/mediapipe/solutions/models.html#pose) | TensorFlow Lite | 1.2.5 and later | [EN](https://medium.com/axinc-ai/blazepose-a-3d-pose-estimation-model-d8689d06b7c4) [JP](https://medium.com/axinc/blazepose-3%E6%AC%A1%E5%85%83%E5%BA%A7%E6%A8%99%E3%82%92%E5%8F%96%E5%BE%97%E5%8F%AF%E8%83%BD%E3%81%AA%E9%AA%A8%E6%A0%BC%E6%A4%9C%E5%87%BA%E3%83%A2%E3%83%87%E3%83%AB-a6c588013009) |
| [<img src="pose_estimation_3d/3dmppe_posenet/output_pose_2d.jpg" width=128px>](pose_estimation_3d/3dmppe_posenet/) |[3dmppe_posenet](/pose_estimation_3d/3dmppe_posenet/) | [PoseNet of "Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image"](https://github.com/mks0601/3DMPPE_POSENET_RELEASE) | Pytorch | 1.2.6 and later | |
| [<img src="pose_estimation_3d/gast/img/output_040.png" width=128px>](pose_estimation_3d/gast/) |[gast](/pose_estimation_3d/gast/) | [A Graph Attention Spatio-temporal Convolutional Networks for 3D Human Pose Estimation in Video (GAST-Net)](https://github.com/fabro66/GAST-Net-3DPoseEstimation) | Pytorch | 1.2.7 and later | [EN](https://medium.com/axinc-ai/gast-a-machine-learning-model-that-predicts-a-3d-skeleton-from-a-2d-skeleton-44449d1ff78d) [JP](https://medium.com/axinc/gast-2d%E3%81%AE%E9%AA%A8%E6%A0%BC%E3%81%8B%E3%82%893d%E3%81%AE%E9%AA%A8%E6%A0%BC%E3%82%92%E4%BA%88%E6%B8%AC%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-6c70fc4b3b3a) |

## Road detection

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="road_detection/codes-for-lane-detection/output.jpg" width=128px>](road_detection/codes-for-lane-detection/) | [codes-for-lane-detection](/road_detection/codes-for-lane-detection/) | [Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection) | Pytorch | 1.2.6 and later | [EN](https://medium.com/axinc-ai/codesforlanedetection-a-machine-learning-model-for-detecting-white-lines-on-roads-7bee3aad818d) [JP](https://medium.com/axinc/codesforlanedetection-%E9%81%93%E8%B7%AF%E3%81%AE%E7%99%BD%E7%B7%9A%E3%82%92%E6%A4%9C%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-1ffe7c6ccf1e) |
| [<img src="road_detection/roneld/output.jpg" width=128px>](road_detection/roneld/) | [roneld](/road_detection/roneld/) | [RONELD-Lane-Detection](https://github.com/czming/RONELD-Lane-Detection) | Pytorch | 1.2.6 and later |
| [<img src="road_detection/road-segmentation-adas/output.png" width=128px>](road_detection/road-segmentation-adas/) | [road-segmentation-adas](/road_detection/road-segmentation-adas/) | [road-segmentation-adas-0001](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/road-segmentation-adas-0001) | OpenVINO | 1.2.5 and later |

## Rotation prediction

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="rotation_prediction/rotnet/output.png" width=256px>](rotation_prediction/rotnet/) |[rotnet](/rotation_prediction/rotnet) | [CNNs for predicting the rotation angle of an image to correct its orientation](https://github.com/d4nst/RotNet) | Keras | 1.2.1 and later |

## Style transfer

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="style_transfer/adain/output.png" width=128px>](style_transfer/adain/) | [adain](/style_transfer/adain/) | [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://github.com/naoto0804/pytorch-AdaIN)| Pytorch | 1.2.1 and later | [EN](https://medium.com/axinc-ai/adain-a-machine-learning-model-for-style-transfer-341b242c554b) [JP](https://medium.com/axinc/adain-%E7%94%BB%E5%83%8F%E3%81%AE%E3%82%B9%E3%82%BF%E3%82%A4%E3%83%AB%E3%82%92%E5%A4%89%E6%8F%9B%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-2443feba832b) |
| [<img src="style_transfer/psgan/output.png" width=128px>](style_transfer/psgan/) | [psgan](/style_transfer/psgan/) | [PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer](https://github.com/wtjiang98/PSGAN)| Pytorch | 1.2.7 and later | |
| [<img src="style_transfer/beauty_gan/output.png" width=128px>](style_transfer/beauty_gan/) | [beauty_gan](/style_transfer/beauty_gan/) | [BeautyGAN](https://github.com/wtjiang98/BeautyGAN_pytorch) | Pytorch | 1.2.7 and later | |
| [<img src="style_transfer/animeganv2/output.png" width=128px>](style_transfer/animeganv2/) | [animeganv2](/style_transfer/animeganv2/) | [PyTorch Implementation of AnimeGANv2](https://github.com/bryandlee/animegan2-pytorch) | Pytorch | 1.2.5 and later | |

## Super resolution

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="super_resolution/srresnet/output.png" width=128px>](super_resolution/srresnet/) | [srresnet](/super_resolution/srresnet/) | [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://github.com/twtygqyy/pytorch-SRResNet) | Pytorch | 1.2.0 and later | [EN](https://medium.com/axinc-ai/srresnet-a-machine-learning-model-to-increase-image-resolution-9efc478f2674) [JP](https://medium.com/axinc/srresnet-%E7%94%BB%E5%83%8F%E3%82%92%E9%AB%98%E5%93%81%E8%B3%AA%E3%81%AB%E6%8B%A1%E5%A4%A7%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-9e35b9a90586) |
| [<img src="super_resolution/edsr/output.png" width=128px>](super_resolution/edsr/) | [edsr](/super_resolution/edsr/) | [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://github.com/sanghyun-son/EDSR-PyTorch.git) | Pytorch | 1.2.6 and later | [EN](https://medium.com/axinc-ai/edsr-a-machine-learning-model-for-super-resolution-image-processing-9deaf36b24ed) [JP](https://medium.com/axinc/edsr-%E7%94%BB%E5%83%8F%E3%81%AE%E8%B6%85%E8%A7%A3%E5%83%8F%E5%87%A6%E7%90%86%E3%82%92%E8%A1%8C%E3%81%86%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-2842b1d244d) |
| [<img src="super_resolution/han/output.png" width=128px>](super_resolution/han/) | [han](/super_resolution/han/) | [Single Image Super-Resolution via a Holistic Attention Network](https://github.com/wwlCape/HAN) | Pytorch | 1.2.6 and later |

## Text detection

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="text_detection/craft_pytorch/imgs_results/res_00_00.jpg" width=64px>](text_detection/craft_pytorch/) |[craft_pytorch](/text_detection/craft_pytorch) | [CRAFT: Character-Region Awareness For Text detection](https://github.com/clovaai/CRAFT-pytorch) | Pytorch | 1.2.2 and later |
| [<img src="text_detection/pixel_link/output.png" width=64px>](text_detection/pixel_link/) |[pixel_link](/text_detection/pixel_link) | [Pixel-Link](https://github.com/ZJULearning/pixel_link) | TensorFlow | 1.2.6 and later |
| [<img src="text_detection/east/output.png" width=64px>](text_detection/east/) |[east](/text_detection/east) | [EAST: An Efficient and Accurate Scene Text Detector](https://github.com/argman/EAST) | TensorFlow | 1.2.6 and later |

## Text recognition

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="text_recognition/etl/font.png" width=64px>](/text_recognition/etl/) |[etl](/text_recognition/etl) | Japanese Character Classification | Keras | 1.1.0 and later | [JP](https://medium.com/axinc/ailia-sdk-%E3%83%A2%E3%83%87%E3%83%AB%E7%B4%B9%E4%BB%8B-etl-412ee389a8ef) |
| [<img src="text_recognition/deep-text-recognition-benchmark/demo_image/demo_1.png" width=64px>](text_recognition/deep-text-recognition-benchmark/) |[deep-text-recognition-benchmark](/text_recognition/deep-text-recognition-benchmark/) | [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) | Pytorch | 1.2.6 and later | |
| [<img src="text_recognition/crnn.pytorch/demo.png" width=64px>](text_recognition/crnn.pytorch/) |[crnn.pytorch](/text_recognition/crnn.pytorch/) | [Convolutional Recurrent Neural Network](https://github.com/meijieru/crnn.pytorch) | Pytorch | 1.2.6 and later | |
| [<img src="text_recognition/paddleocr/output.png" width=64px>](text_recognition/paddleocr/) |[paddleocr](/text_recognition/paddleocr/) | [PaddleOCR : Awesome multilingual OCR toolkits based on PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR) | Pytorch | 1.2.6 and later | [EN](https://medium.com/axinc-ai/paddleocr-the-latest-lightweight-ocr-system-a13171d7ea3e) [JP](https://medium.com/axinc/paddleocr-%E6%9C%80%E6%96%B0%E3%81%AE%E8%BB%BD%E9%87%8Focr%E3%82%B7%E3%82%B9%E3%83%86%E3%83%A0-8744205f3703) |

## Vehicle recognition

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="vehicle_recognition/vehicle-attributes-recognition-barrier/demo.png" width=64px>](/vehicle_recognition/vehicle-attributes-recognition-barrier/) |[vehicle-attributes-recognition-barrier](/vehicle_recognition/vehicle-attributes-recognition-barrier) | [vehicle-attributes-recognition-barrier-0042](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-attributes-recognition-barrier-0042) | OpenVINO | 1.2.5 and later | [EN](https://medium.com/axinc-ai/vehicleattributerecognitionbarrier-a-machine-learning-model-for-detecting-car-attributes-fe8fda7649ff) [JP](https://medium.com/axinc/vehicleattributerecognitionbarrier-%E8%BB%8A%E3%81%AE%E5%B1%9E%E6%80%A7%E3%82%92%E6%A4%9C%E5%87%BA%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-ee26d1a3e00b) |
| [<img src="vehicle_recognition/vehicle-license-plate-detection-barrier/output.png" width=128px>](vehicle_recognition/vehicle-license-plate-detection-barrier/) | [vehicle-license-plate-detection-barrier](/vehicle_recognition/vehicle-license-plate-detection-barrier/) | [vehicle-license-plate-detection-barrier-0106](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-license-plate-detection-barrier-0106) | OpenVINO | 1.2.5 and later | |

## Commercial model

| Model | Reference | Exported From | Supported Ailia Version | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|
|[acculus-pose](/commercial_model/acculus-pose) | [Acculus, Inc.](https://acculus.jp/) | Caffe | 1.2.3 and later |

# Other languages

[unity version](https://github.com/axinc-ai/ailia-models-unity)

[c++ version](https://github.com/axinc-ai/ailia-models-cpp)
