# crnn-audio-classification

### input
wave file path

### output
labels

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

[crnn.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/crnn/crnn.onnx.prototxt)
