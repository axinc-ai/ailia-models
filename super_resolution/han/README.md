# HAN

## Input

|            Input           |  Ailia input shape |    Range   | 
| :------------------------: | :----------------: | :--------: | 
| ![](images/000002_LR.png)  |    (1,3,194,194)   | [0.0, 1.0] |
| ![](images/lenna.png)      |     (1,3,64,64)    | [0.0, 1.0] |

## Output

| Resolution scale (Degradation model) |             x2 (BI)            |             x3  (BI)           |             x4 (BI)            |             x8 (BI)           |             x3 (BD)            |
| :----------------------------------: | :----------------------------: | :----------------------------: | :----------------------------: | :---------------------: | :----------------------------: | 
|                 Output               | ![](images/000002_LR_BIX2.png) | ![](images/000002_LR_BIX3.png) | ![](images/000002_LR_BIX4.png) | ![](000002_LR_BIX8.png) | ![](images/000002_LR_BDX3.png) |
|           Ailia output shape         |           (1,3,388,388)        |           (1,3,582,582)        |           (1,3,776,776)        |          (1,3,1552,1552)      |          (1,3,582,582)         |
|                  Range               |            [0.0, 1.0]          |            [0.0, 1.0]          |            [0.0, 1.0]          |            [0.0, 1.0]         |           [0.0, 1.0]           |

| Resolution scale (Degradation model) |           x2 (BI)          |            x3  (BI)        |            x4 (BI)         |           x8 (BI)          |            x3 (BD)       |
| :----------------------------------: | :------------------------: | :------------------------: | :------------------------: | :------------------------: | :------------------------: | 
|                 Output               | ![](images/lenna_BIX2.png) | ![](images/lenna_BIX3.png) | ![](images/lenna_BIX4.png) | ![](images/lenna_BIX8.png) | ![](images/lenna_BDX3.png) |
|           Ailia output shape         |        (1,3,128,128)       |         (1,3,192,192)      |         (1,3,256,256)      |         (1,3,512,512)      |         (1,3,192,192)       |
|                  Range               |          [0.0, 1.0]        |           [0.0, 1.0]       |           [0.0, 1.0]       |          [0.0, 1.0]        |          [0.0, 1.0]         |

## Usage
Automatically downloads the onnx and prototxt files when running.
It is necessary to be connected to the Internet while downloading.

For the sample image with twice the resolution (BI),
``` bash
$ python3 han.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 han.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

If you want to specify the scale for the resolution, put the scale after the `--scale` option.  
Choose the scale in [2, 3, 4, 8].
```bash
$ python3 han.py --scale SCALE 
```

If you want to the model trained on imaged degraded by the Blur-downscale Degradation Model, specify the `--blur` option.  
Only a 3-resolution scale can be used with this option. 
```bash
$ python3 han.py --scale 3 --blur 
```

The default setting is to use the optimized model and weights, but you can also switch to the normal model by using the `--normal` option.
If the output image is entirely black, try to add the `-e 0` option.
``` bash
$ python3 han.py -e 0
```

## Reference

[Single Image Super-Resolution via a Holistic Attention Network](https://github.com/wwlCape/HAN.git)

## Framework

Pytorch 1.3.0

## Model Format

ONNX opset = 11

## Netron

[han_BIX2.onnx.prototxt](https://storage.googleapis.com/ailia-models/han/han_BIX2.onnx.prototxt)
[han_BIX2.opt.onnx.prototxt](https://storage.googleapis.com/ailia-models/han/han_BIX2.opt.onnx.prototxt)

[han_BIX3.onnx.prototxt](https://storage.googleapis.com/ailia-models/han/han_BIX3.onnx.prototxt)
[han_BIX3.opt.onnx.prototxt](https://storage.googleapis.com/ailia-models/han/han_BIX3.opt.onnx.prototxt)

[han_BIX4.onnx.prototxt](https://storage.googleapis.com/ailia-models/han/han_BIX4.onnx.prototxt)
[han_BIX4.opt.onnx.prototxt](https://storage.googleapis.com/ailia-models/han/han_BIX4.opt.onnx.prototxt)

[han_BIX8.onnx.prototxt](https://storage.googleapis.com/ailia-models/han/han_BIX8.onnx.prototxt)
[han_BIX8.opt.onnx.prototxt](https://storage.googleapis.com/ailia-models/han/han_BIX8.opt.onnx.prototxt)

[han_BDX3.onnx.prototxt](https://storage.googleapis.com/ailia-models/han/han_BDX3.onnx.prototxt)
[han_BDX3.opt.onnx.prototxt](https://storage.googleapis.com/ailia-models/han/han_BDX3.opt.onnx.prototxt)
