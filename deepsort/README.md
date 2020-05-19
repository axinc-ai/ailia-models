# DeepSort

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample video,
``` bash
$ python3 deepsort.py
```

If you want to specify the input video, put the video path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 deepsort.py --input VIDEO_PATH --savepath SAVE_IMAGE_PATH
```

### Reference

[Deep Sort with PyTorch](https://github.com/ZQPei/deep_sort_pytorch)


### Framework
PyTorch 0.4


### Model Format
ONNX opset = 10


### Netron
[deep_sort.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/deep_sort/deep_sort.onnx.prototxt)
