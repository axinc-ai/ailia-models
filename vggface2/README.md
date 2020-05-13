# VGGFace2 Dataset for Face Recognition

## Input

- image_a : ![Input](couple_a.jpg)
- image_b : ![Input](couple_b.jpg)
- image_c : ![Input](couple_c.jpg)

Ailia input shape : (1, 3, 224, 224)


## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

By default, the following two images are loaded: `couple_a.jpg`, `couple_c.jpg` (Same person example)

```bash
$ python3 vggface2.py
...
couple_a.jpg vs couple_c.jpg = 0.6718782782554626
Same person
```

If you want to specify images, speficy the paths of the two images after the `--inputs` option.
```bash
$ python3 vggface2.py --inputs IMAGE_PATH1 IMAGE_PATH2
```

By adding `--video` option, you can compare the face from video frame and the still image, and calculate the distance.
If you pass `0` as an argument to `VIDEO_PATH`, you can use the webcam input instead of the video file.
```bash
$ python3 vggface2.py --video VIDEO_PATH IMAGE_PATH
$ python3 vggface2.py --video 0 IMAGE_PATH
```

## Reference

- [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- [VGGFace2 Dataset for Face Recognition](https://github.com/ox-vgg/vgg_face2)

## Framework

Caffe

## Model Format

CaffeModel

## Netron

[resnet50_scratch.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/vggface2/resnet50_scratch.prototxt)
