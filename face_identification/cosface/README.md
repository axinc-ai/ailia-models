# CosFace

### Input
| Image filename   | Image                       |
|------------------|-----------------------------|
 | image_id.jpg     | ![Input](image_id.jpg)      |
 | image_target.jpg | ![Input](image_target.jpg)] |
(Image from https://github.com/ronghuaiyang/arcface-pytorch/issues/63)
 - Ailia input shape : (1, 3, 96, 112)
 - Range: [-1.0,1.0]

### Output
 A similarity of a pair of images.

### Usage
Automatically downloads the onnx and onnx file on the first run.  
It is necessary to be connected to the Internet while downloading.

By default, the following two images are loaded: `image_id.jpg`, `image_target.jpg` 
``` bash
$ python3 acosface.py
...
Similarity of (image_id.jpg, image_target.jpg) : 0.209
Script finished successfully.
```

If you want to specify images, specify the paths of the two images after the `--inputs` option.
``` bash
$ python3 cosface.py --inputs IMAGE_PATH1 IMAGE_PATH2
```

### Paper
 - CosFace: Large Margin Cosine Loss for Deep Face Recognition
   - https://arxiv.org/abs/1801.09414

### Reference
 - https://github.com/MuggleWang/CosFace_pytorch

### Dependencies
  - requirements.txt

### Model Format
  - ONNX opset = 10