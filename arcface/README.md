# ArcFace

### input

![correct_pair_1_image](correct_pair_1.jpg)
![correct_pair_2_image](correct_pair_2.jpg)
![incorrect_iamge](incorrect.jpg)

(Image from https://github.com/ronghuaiyang/arcface-pytorch/issues/63)

Shape(4, 1, 128, 128) Range:[-1, 1]  (original image & flipped iamge for a pair image)

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

If you want to compare a webcam captured image with a certain image, specify the base image to be compared after the --camera option.

If a webcam is detected, a window will open automatically.  
The area enclosed by the green square is the part that will actually be loaded into the model.  
So, adjust the camera so that the face fits in the green square.
```bash
$ python3 arcface.py --camera BASE_IMAGE_PATH
```



### Reference
[arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)


### Framework
PyTorch 1.3.1


### Model Format
ONNX opset = 10


### Netron
[arcface.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/arcface/arcface.onnx.prototxt)
