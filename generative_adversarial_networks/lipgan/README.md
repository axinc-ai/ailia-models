# lipgan : Generate realistic talking faces for any human speech and face identity.

## Input

[<img src="input.jpg" width=256px>](input.jpg)

[input.wav](innput.wav)

## Output

mp4

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 lipgan.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
If you want to specify the input audio, put the audio path after the `--audio` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 lipgan.py --input IMAGE_PATH --audio AUDIO_PATH --savepath SAVE_VIDEO_PATH 
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 lipgan.py --video VIDEO_PATH 
```

By adding the `--use_dlib` option, you can use original version of face alignment.

By adding the `--merge_audio` option, you can merge output video and input audio using ffmpeg.

By adding the `--ailia_audio` option, you can use ailia audio library for melspectrum conversion.

By adding the `--realesrgan` option, you can use super resoltuion for geneerated face image.

## Reference

- [LipGAN](https://github.com/Rudrabha/LipGAN)

## Framework

- Keras
- tensorflow2onnx

## Model Format

ONNX opset=15

## Netron

[lipgan.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/lipgam/lipgan.onnx.prototxt)
