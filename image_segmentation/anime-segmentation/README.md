# Anime Segmentation

### Input

![input_image](demo.png)  

(Image from https://datasets-server.huggingface.co/assets/skytnt/anime-segmentation/--/imgs-masks/train/51/image/image.jpg)

### Output
![output_image](output.png)

### Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 anime-segmentation.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 anime-segmentation.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 anime-segmentation.py --video VIDEO_PATH
```

### Reference

- [Anime Segmentation](https://github.com/SkyTNT/anime-segmentation)
- [Hugging Face - skytnt/anime-segmentation](https://huggingface.co/datasets/skytnt/anime-segmentation)

### Framework

PyTorch

### Model Format

ONNX opset = 11

### Netron

[isnetis.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/anime-segmentation/isnetis.onnx.prototxt)
