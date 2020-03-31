# ArcFace

### input

![correct_pair_1_image](correct_pair_1.jpg)
![correct_pair_2_image](correct_pair_2.jpg)
![incorrect_iamge](incorrect.jpg)

(Image from https://github.com/ronghuaiyang/arcface-pytorch/issues/63)

Ailia input Shape(4, 1, 128, 128) Range:[-1, 1]  (original image & flipped iamge for a pair image)


### output
A similarity of a pair of images.


### usage
Automatically downloads the onnx and prototxt files on the first run.  
It is necessary to be connected to the Internet while downloading.

By default, the following two images are loaded: `correct_pair_1.jpg`, `correct_pair_2.jpg`
``` bash
$ python3 arcface.py
Similarity of (correct_pair_1.jpg, correct_pair_2.jpg) : 0.5981666445732117
```

If you want to specify images, specify the paths of the two images after the --input option.
``` bash
$ python3 arcface.py --input IMAGE_PATH1 IMAGE_PATH2
```

By adding the VIDEO option, you can compare the face of the video and the still image
and calculate the similarity.
If you pass 0 as an argument to VIDEO_PATH, you can use the webcam input instead of the video.
```bash
$ python3 arcface.py --video VIDEO_PATH IMAGE_PATH

$ python3 arcface.py --video 0 IMAGE_PATH
```


### Reference
[arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)


### Framework
PyTorch 1.3.1


### Model Format
ONNX opset = 10


### Netron
[arcface.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/arcface/arcface.onnx.prototxt)
