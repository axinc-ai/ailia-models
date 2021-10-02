# RAFT: Recurrent All Pairs Field Transforms for Optical Flow

## input

（1 frame before）<br/>
<img src="https://user-images.githubusercontent.com/9916906/135715920-3f3db27c-9efc-4199-924b-4f3ae99455f5.png" alt="drawing" width="300"/>

（1 frame after）<br/>
<img src="https://user-images.githubusercontent.com/9916906/135715898-4a34ff94-80f1-40fd-b9d4-9d9bf4f3e9a6.png" alt="drawing" width="300"/>

(from https://pixabay.com/videos/car-racing-motor-sports-action-74/)

<br/>

## output

（estimated optical flow）<br/>
<img src="https://user-images.githubusercontent.com/9916906/135715855-28781f4a-1b11-4032-a785-7cbf9be0cced.png" alt="drawing" width="300"/>

Color is an estimation of the direction in which the object moved between frames.<br/>
<img src="https://user-images.githubusercontent.com/9916906/135722785-b14895af-9434-4a6b-90d1-6e72be3e3222.png" alt="drawing" width="150"/>

<br/>

（video's output）<br/>
<img src="https://user-images.githubusercontent.com/9916906/135719449-a5546878-5cc4-431c-8d5f-c8ba33e88929.gif" alt="drawing" width="300"/>

<br/>

## usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python raft.py
(ex on CPU)  $ python raft.py -e 0
(ex on BLAS) $ python raft.py -e 1
(ex on GPU)  $ python raft.py -e 2
```

If you want to specify the input images, put the two images path after the `--inputs` option.
Specify the frame images in the video in the order of front and back.
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 raft.py --inputs IMAGE_PATH_BEFORE_FRAME IMAGE_PATH_AFTER_FRAME --savepath SAVE_IMAGE_PATH
$ python3 raft.py -i IMAGE_PATH_BEFORE_FRAME IMAGE_PATH_AFTER_FRAME -s SAVE_IMAGE_PATH
(ex) $ python3 raft.py --inputs input_before.png input_after.png --savepath output.png
```

By adding the `--video` option, you can input the video.
```bash
$ python3 raft.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
$ python3 raft.py -v VIDEO_PATH -s SAVE_VIDEO_PATH
(ex) $ python3 raft.py --video input.mp4 --savepath output.mp4
```

By the way, if the input data has a high resolution then the accuracy tends to be high, and if the input data has a low resolution then the processing speed tends to be high.

<br/>

## Reference

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow (https://github.com/princeton-vl/RAFT)

<br/>

## Framework
Pytorch

<br/>

## Model Format
ONNX opset = 11

<br/>

## Netron

[raft-things_fnet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/raft/raft-things_fnet.onnx.prototxt)<br/>
[raft-things_cnet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/raft/raft-things_cnet.onnx.prototxt)<br/>
[raft-things_update_block.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/raft/raft-things_update_block.onnx.prototxt)<br/>
[raft-small_fnet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/raft/raft-small_fnet.onnx.prototxt)<br/>
[raft-small_cnet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/raft/raft-small_cnet.onnx.prototxt)<br/>
[raft-small_update_block.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/raft/raft-small_update_block.onnx.prototxt)<br/>
