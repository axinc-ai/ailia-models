# ArcFace

### input

![correct_pair_1_image](correct_pair_1.jpg)
![correct_pair_2_image](correct_pair_2.jpg)
![incorrect_iamge](incorrect.jpg)

(Image from https://github.com/ronghuaiyang/arcface-pytorch/issues/63)

Shape(1, 1, 128, 128) Range:[-1, 1]

### output
A similarity of a pair of images.


### usage

``` bash
$ python3 arcface.py
Similarity of (correct_pair_1.jpg, correct_pair_2.jpg) : 0.5981666445732117

$ python3 arcface.py
Similarity of (correct_pair_1.jpg, incorrect.jpg) : -0.010565089993178844
```

Change `IMG_PATH_1` and `IMG_PATH_2` in `arcface.py`.
We give two images at once as a batch, then calculate the similarity of these two images based on the model output.


### Reference
[arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)


### Framework
PyTorch 1.3.1


### Model Format
ONNX opset = 10


### Netron
[arcface.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/arcface/arcface.onnx.prototxt)
