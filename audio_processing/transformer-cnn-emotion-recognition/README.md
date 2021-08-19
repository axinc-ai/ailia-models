# transformer-cnn-emotion-recognition

## Input

audio file

```
03-01-01-01-01-01-01.wav
RAVDESS Dataset
https://smartlaboratory.org/ravdess/
```

## Output

emotion label

```
Emotion: neutral
Confidence: 0.99993193
```

## Labels

```
"surprised", "neutral", "calm", "happy",
"sad", "angry", "fearful", "disgust"
```

## Requirements
This model requires additional module.

```
pip3 install librosa
```

## Usage

```bash
$ python3 transformer-cnn-emotion-recognition.py -i input.wav
```

## Reference

[Combining Spatial and Temporal Feature Representions of Speech Emotion by Parallelizing CNNs and Transformer-Encoders](https://github.com/IliaZenkov/transformer-cnn-emotion-recognition)

## Framework

PyTorch 1.6.0

## Model Format

ONNX opset = 11

## Netron

[parallel_is_all_you_want_ep428.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/parallel_is_all_you_want/parallel_is_all_you_want_ep428.onnx.prototxt)
