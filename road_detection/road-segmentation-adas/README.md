# road-segmentation-adas-0001

## Input

![Input](demo.png)

(Image from https://www.pexels.com/ja-jp/video/854669/)

Shape : (1, 512, 896, 3) BGR channel order

## Output

![Output](output.png)

Shape : (1, 512, 896, 4)

### Category

```
CATEGORY = {
    'BG': 0,
    'road': 1,
    'curb': 2,
    'mark': 3,
}
```

### Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 road-segmentation-adas.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 road-segmentation-adas.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 road-segmentation-adas.py --video VIDEO_PATH
```

## Reference

- [OpenVINO - Open Model Zoo repository - road-segmentation-adas-0001](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/road-segmentation-adas-0001)
- [OpenVINO - road-segmentation-adas-0001](https://docs.openvinotoolkit.org/latest/omz_models_model_road_segmentation_adas_0001.html)

## Framework

OpenVINO

## Model Format

ONNX opset=11

## Netron

[road-segmentation-adas-0001.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/road-segmentation-adas/road-segmentation-adas-0001.onnx.prototxt)  
