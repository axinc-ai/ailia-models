# MonoDepth2

### input

![input_image](input.jpg)

(Image from https://github.com/nianticlabs/monodepth2/tree/master/assets/test_image.jpg)

Shape(1, 3, 192, 640) Range:[0, 1]

### output

![output_image](output.png)

### Note

This Software is licensed under the terms of the following Monodepth2 license
which allows for non-commercial use only. For any other use of the software not
covered by the terms of this license, please contact partnerships@nianticlabs.com

### usage

``` bash
python3 monodepth2.py
```
you can change input image path in `monodepth2.py`


### Reference

[Monocular depth estimation from a single image](https://github.com/nianticlabs/monodepth2)


### Framework
PyTorch 0.4.1


### Model Format
ONNX opset = 10


### Netron

[monodepth2_mono+stereo_640x192_enc.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/monodepth2/monodepth2_mono%2Bstereo_640x192_enc.onnx.prototxt)
[monodepth2_mono+stereo_640x192_dec.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/monodepth2/monodepth2_mono%2Bstereo_640x192_dec.onnx.prototxt)