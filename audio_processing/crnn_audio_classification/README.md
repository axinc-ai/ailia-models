# crnn_audio_classification

### input

audio file

```
24965__www-bonson-ca__bigdogbarking-02.wav
Attribution 3.0 Unported (CC BY 3.0)
https://freesound.org/people/www.bonson.ca/sounds/24965/
```

### output

audio label

```
dog_bark
0.8683825731277466
```

### labels

```
'air_conditioner',
'car_horn',
'children_playing',
'dog_bark',
'drilling',
'engine_idling',
'gun_shot',
'jackhammer',
'siren',
'street_music'
```

### Usage

```bash
$ python3 crnn-audio-classification.py -i input.wav
```

### Reference
[crnn-audio-classification](https://github.com/ksanjeevan/crnn-audio-classification)  

### Framework
PyTorch 1.6.0

### Model Format
ONNX opset = 10

### Netron

[crnn.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/crnn/crnn.onnx.prototxt)
