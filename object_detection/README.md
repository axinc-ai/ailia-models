# ailia MODELS : Object Detection

<img src="./yolox/output.jpg">

## Model detail

Detect bounding box of objects from single image.

## Model metrics

|Name|mAP50|GFlops|Resolution|Publish Date|
|-----|-----|-----|-----|-----|
|[yolox_l](./yolox/)||155.6|640|2021.8|
|[yolov3](./yolov3/)|65.23|65.86|416|2018.4|
|[yolov4](./yolov4/)|64.38|129.5|416|2020.4|
|[yolox_m](./yolox/)|62.12|73.8|640|2021.8|
|[yolox_s](./yolox/)|56.35|26.8|640|2021.8|
|[yolov5s6](./yolov5/)|22.44 (error?)|16.8|640|2021.10|
|[yolov5s](./yolov5/)||16.5|640|2020.6|
|[yolox_tiny](./yolox/)|47.04|6.45|416|2021.8|
|[yolox_nano](./yolox/)|39.03|1.08|416|2021.8|
|[yolov4_tiny](./yolov4-tiny/)|36.31|6.92|416|2020.4|
|[yolov3_tiny](./yolov3-tiny/)|35.76|5.56|416|2018.4|

## mAP (Accuracy)

Basically the accuracy of object detection algorithm is calculated by mAP. In this page, mAP was calculated using this repository.

https://github.com/rafaelpadilla/Object-Detection-Metrics

We used COCO2017 val images for testing. We set parameters, iou = 0.5 (mAP50) and detection threshold = 0.01 (Because small value achieves high accuracy).

## GFlops (Computing cost)

GFlops was referred from below site.

- yolox : https://github.com/Megvii-BaseDetection/YOLOX
- yolov5 : https://github.com/ultralytics/yolov5
- yolov4 : https://github.com/Tianxiaomo/pytorch-YOLOv4 https://docs.openvino.ai/latest/omz_models_model_yolo_v4_tiny_tf.html
- yolov3 : https://pjreddie.com/darknet/yolo/
