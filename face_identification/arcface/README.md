# ArcFace

### input
- correct pair  
![correct_pair_1_image](correct_pair_1.jpg)
![correct_pair_2_image](correct_pair_2.jpg)

- incorrect pair  
![correct_pair_1_image](correct_pair_1.jpg)
![incorrect_iamge](incorrect.jpg)

(Image from https://github.com/ronghuaiyang/arcface-pytorch/issues/63)

Input the original image1 and its inversion, and the original image2 and its inversion.  
Invered images are generated automatically.  
(All images are treated as grayscale images)
- Ailia input Shape: (4, 1, 128, 128)  
- Range: [-1, 1]  


### output
A similarity of a pair of images.


### usage
Automatically downloads the onnx and prototxt files on the first run.  
It is necessary to be connected to the Internet while downloading.

By default, the following two images are loaded: `correct_pair_1.jpg`, `correct_pair_2.jpg`
``` bash
$ python3 arcface.py
...
Similarity of (correct_pair_1.jpg, correct_pair_2.jpg) : 0.5981666445732117
They are the same face!
```

If you want to specify images, specify the paths of the two images after the `--inputs` option.
``` bash
$ python3 arcface.py --inputs IMAGE_PATH1 IMAGE_PATH2
```

By adding the `--video` option, you can compare the face of the video
and calculate the similarity.
If you pass `0` as an argument to `VIDEO_PATH`, you can use the webcam input instead of the video.
```bash
$ python3 arcface.py --video VIDEO_PATH

$ python3 arcface.py --video 0
```


### Reference
[arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)


### Framework
PyTorch 1.3.1


### Model Format
ONNX opset = 10


### Netron
[arcface.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/arcface/arcface.onnx.prototxt)
