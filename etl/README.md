# Japanese Character Classification

### input

![input_image](font.png)

Ailia input shape: (1, 1, 28, 28)  
Range: [0, 1]

### output

[3189 character](etl_BINARY_squeezenet128_20.txt)

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 blazeface.py 

+ idx=0
0
3151
  category=0[ あ
 ]
  prob=0.9986122846603394
+ idx=1
4
3151
  category=4[ お
 ]
  prob=0.0010359695879742503
+ idx=2
59
3151
  category=59[ め
 ]
  prob=0.0003514652489684522
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 etl.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 etl.py --video VIDEO_PATH
```

### Dataset

[ETL Dataset](http://etlcdb.db.aist.go.jp/?lang=ja)

### Framework

Keras

### Model Format

CaffeModel

### Netron

[etl_BINARY_squeezenet128_20.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/etl/etl_BINARY_squeezenet128_20.prototxt)
