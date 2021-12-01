# SiamMOT

## Input

![Input](input.png)

(Video from https://vimeo.com/60139361)

## Output

![Output](output.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample video,
```bash
$ python3 siam-mot.py
```

If you want to specify the video image, put the file path after the `--video` option.  
You can use `--savepath` option to specify the name of the output file to save.
```bash
$ python3 siam-mot.py --video VIDEO_PATH --savepath SAVE_FILE_PATH
```

You can choose `person` or `person_vehicle` for track-class such that person tracking or person/vehicle tracking model is used accordingly.
```bash
$ python3 siam-mot.py --track-class person
```

## Reference

- [SiamMOT](https://github.com/amazon-research/siam-mot)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[person_rpn.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siam-mot/person_rpn.onnx.prototxt)  
[person_box.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siam-mot/person_box.onnx.prototxt)  
[person_track.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siam-mot/person_track.onnx.prototxt)  
[person_feat_ext.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siam-mot/person_feat_ext.onnx.prototxt)  
[person_vehicle_rpn.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siam-mot/person_vehicle_rpn.onnx.prototxt)  
[person_vehicle_box.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siam-mot/person_vehicle_box.onnx.prototxt)  
[person_vehicle_track.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siam-mot/person_vehicle_track.onnx.prototxt)  
[person_vehicle_feat_ext.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siam-mot/person_vehicle_feat_ext.onnx.prototxt)
