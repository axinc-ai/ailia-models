# Weather Prediction From Image - (Warmth Of Image)


![](https://user-images.githubusercontent.com/20780894/35770726-967a4198-0931-11e8-93a2-c6b8cb826210.jpg)

**Project Topic:** Weather Condition Prediction from Image

**_Abstract:_** Basically, problem is recognise or predict weather condition(air condition) in given picture(image data) by using extracted useful features from image and advanced methods/algorithms. Enlarging previous works and covering different label types are different point of the research.


**Project Members:** 
- [Berk Gulay](https://www.linkedin.com/in/berk-gulay97/)
- [Samet Kalkan](https://www.linkedin.com/in/sametkalkan/)
- [Mert Surucuoglu](https://www.linkedin.com/in/mertsurucu/)

**Institution-Department:** Hacettepe University - Computer Science

**You can find links to dataset of the work and proper explanations in pinned issues :)**
You're very welcome to check introductory informations and details of the project by visiting links below :)

[Project's Final Report -->](https://drive.google.com/open?id=1HFyAUvnkS61Xat9cUBAhG-4hvwR1T8lb)

[Short Video Presentation -->](https://www.youtube.com/watch?v=TdzUGoS2F80&t=7s)

[Medium Blog -->](https://medium.com/warmthofimage)

[Android Application -->](https://play.google.com/store/apps/details?id=com.kalkan.weatherprediction)


**DATASET:**

[Class 0 -->](https://drive.google.com/open?id=1j9nLFoAA5xxC5DplQd-mgbysVsMcJEcz)

[Class 1 -->](https://drive.google.com/open?id=1KO0ryOH6j4pFYTJpSyGxL1iH1f84dNU7)

[Class 2,3,4 -->](https://drive.google.com/open?id=1MWLGbv82_aEZo3h84pQAMRHkvCvsWSkA)

## ailia
### Input
- input image (1x3x250x250)

<img src="./data/img/00001.png" width="240px">

(Extracted from [the dataset above](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).)

### Output
- output image (1x1x224x224)

<img src="./output.png" width="240px">

### Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 weather-prediction-from-image.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the directory of the output file to be saved.
```bash
$ python3 weather-prediction-from-image.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video and convert it by the style image.
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 weather-prediction-from-image.py --video VIDEO_PATH
```

### Framework
keras

### Model Format
ONNX opset = 13

### Netron
- [weather-prediction-from-image_trainedModelE20.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/weather-prediction-from-image/weather-prediction-from-image_trainedModelE20.onnx.prototxt)
