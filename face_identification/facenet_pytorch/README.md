# Face Recognition Using Pytorch

## Input
3 face images.
<figure>
<img src="data\angelina_jolie.jpg" width="150">
<img src="data\bradley_cooper.jpg" width="150">
<img src="data\kate_siegel.jpg" width="150">
<legend>angelina_jolie.jpg 　　　bradley_cooper.jpg　　　kate_siegel.jpg</legend>
</figure>

(Image from https://github.com/timesler/facenet-pytorch/tree/master/data/test_images)

## Output

A similarity of each pair of images.
```
Similarity of ('angelina_jolie.jpg', 'kate_siegel.jpg') is 0.891.
Similarity of ('angelina_jolie.jpg', 'bradley_cooper.jpg') is 1.446.
Similarity of ('kate_siegel.jpg', 'bradley_cooper.jpg') is 1.302.
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample images,
```bash
$ python3 facenet_pytorch.py
```

If you want to specify the image folder, put the file path after the `-d (--dir)` option.  
```bash
$ python3 facenet_pytorch.py -d IMG_DIR_PATH
```
`-w (--weight)` option select trained weights with the following datasets.

- vggface2
- casia-webface
```bash
$ python3 facenet_pytorch.py -w casia-webface
```

## Reference

- [facenet_pytorch](https://github.com/timesler/facenet-pytorch)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron
 - [facenet](https://netron.app/?url=https://storage.googleapis.com/ailia-models/facenet-pytorch/vggface2.onnx.prototxt)
