# Illustration2Vec

### input
![input_image](input.jpg)  
Hatsune Miku (初音ミク), © Crypton Future Media, INC., http://piapro.net/en_for_creators.html. This image is licensed under the Creative Commons - Attribution-NonCommercial, 3.0 Unported (CC BY-NC).  
- Ailia input shape: (1, 3, 224, 224)  

### output
- Estimating Tag Mode
```bash
[{'character': [('hatsune miku', 0.9999994039535522)],
  'copyright': [('vocaloid', 0.9999998807907104)],
  'general': [('thighhighs', 0.9957002997398376),
              ('1girl', 0.9873842000961304),
              ('twintails', 0.9814184308052063),
              ('solo', 0.9636198878288269),
              ('aqua hair', 0.9165698885917664),
              ('long hair', 0.8823321461677551),
              ('very long hair', 0.8341774940490723),
              ('detached sleeves', 0.7452000975608826),
              ('skirt', 0.6787945628166199),
              ('necktie', 0.5615589618682861),
              ('aqua eyes', 0.552979588508606)],
  'rating': [('safe', 0.9787901043891907),
             ('questionable', 0.020344465970993042),
             ('explicit', 0.0006209250423125923)]}]
```

- Extracting Feature Vector Mode
```bash
[[ 7.4673967   3.6940672   0.53971916 ... -0.08819806  2.720264
   7.3380337 ]]
```

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 illustration2vec.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 illustration2vec.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 illustration2vec.py --video VIDEO_PATH
```

The default mode is to predict the tag, but you can switch to the mode to extract the feature vector
by specifying `--featurevec`.
```bash
$ python3 illustration2vec --featurevec
```

### Reference
- [Illustration2Vec](https://github.com/rezoo/illustration2vec)
- [Illustration2Vec for ONNX](https://github.com/kivantium/illustration2vec)


### Framework
Caffe


### Model Format
ONNX opset = 10


### Netron

- [illust2vec_ver200.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/ill2vec/illust2vec_ver200.onnx.prototxt)
- [illust2vec_tag_ver200.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/ill2vec/illust2vec_tag_ver200.onnx.prototxt)
