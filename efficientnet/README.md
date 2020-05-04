# EfficientNet

## Input

![Input](teddy_bear_resize.jpg)

Ailia input shape : (1,3,224,224)  
Range : [-1.0, 1.0]

## Output

```
+ idx=0
  category=850[ teddy, teddy bear ]
  prob=8.607271194458008
+ idx=1
  category=865[ toyshop ]
  prob=2.786747932434082
+ idx=2
  category=520[ crib, cot ]
  prob=2.3403470516204834
```

### usage

For the sample image,
``` bash
$ python3 efficientnet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 efficientnet.py --input IMAGE_PATH
```

## Reference

[A PyTorch implementation of EfficientNet]( https://github.com/lukemelas/EfficientNet-PyTorch)

## Model Format

ONNX opset = 10

## Framework

Pytorch 1.1.0
