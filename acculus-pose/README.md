# acculus-pose

## Notice

This model is a commercial model. Please contact contact@axinc.jp to get the model.

## Usage

For the sample image,
``` bash
$ python3 lightweight-human-pose-estimation.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 acculus-pose.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
$ python3 acculus-up-pose.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 acculus-pose.py --video VIDEO_PATH
$ python3 acculus-up-pose.py --video VIDEO_PATH
$ python3 acculus-hand.py --video VIDEO_PATH
```

## Reference

[Acculus, Inc.](https://acculus.jp/)

## Model Format

CaffeModel
