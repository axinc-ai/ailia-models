# BlazeFace

### input

<img src="input.png" width="320px">

(Image from https://github.com/hollance/BlazeFace-PyTorch/blob/master/3faces.png)

Shape(1, 3, 128, 128) Range:[-1, 1]

### output

![output_image](result.png)

### usage

``` bash
python3 blazeface.py 
```
you can change input image path in `blazeface.py`

``` bash
python3 blazeface.py video
```
you can use webcamera input

### Reference

[BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch)


### Framework
PyTorch 1.1


### Model Format
ONNX opset = 10


### Netron

[blazeface.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/blazeface/blazeface.onnx.prototxt)
