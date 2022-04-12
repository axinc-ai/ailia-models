# EgoNet

## Input

![Input](007161.png)

(Image from http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

## Output

![Output](output.png)

## Data Preparation

You need to download KITTI dataset [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Download left images, calibration files and labels. Also download the [D4LCN.zip](https://drive.google.com/drive/folders/1atfXLmsLFG6XEtNnwZuEYLydKqjr7Icf?usp=sharing) (predicted).  
Data folder should look like this:
```
egonet
├── calib
  ├── xxx.txt (Camera parameters for image xxx: provided from data_object_calib.zip)
├── label
  ├── xxx.txt (predicted object labels for image xxx: provided from D4LCN.zip)
├── gt_label
  ├── xxx.txt (ground-truth object labels for image xxx: provided from data_object_calib.zip)
```

These files are written in kitti format.

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 egonet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 egonet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

You can specify directory or directly file path with the `--label_path`, `--gt_label_path`
and `--calib_path` options for find label, gt_label, calib file.
```bash
$ python3 egonet.py --label_path LABEL_PATH --gt_label_path GT_LABEL_PATH --calib_path CALIB_PATH
```

You can get the 3D plot by specifying the `--plot_3d` option.
```bash
$ python3 egonet.py --plot_3d
```

For images without a label file, you can use the `--detector` option to get the BBOX.
```bash
$ python3 egonet.py --detector
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 egonet.py --video VIDEO_PATH
```

## Reference

- [EgoNet](https://github.com/Nicholasli1995/EgoNet)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[HC.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/egonet/HC.onnx.prototxt)  
[L.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/egonet/L.onnx.prototxt)
