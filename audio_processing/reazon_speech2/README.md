# ReazonSpeech v2.0

## Input

Audio file

https://user-images.githubusercontent.com/29946532/218316186-266ffd00-4b1d-42c1-aa45-0080d781feb5.mov

## Output

Recognized speech text
```
[00:00:00.300 --> 00:00:04.780] 気象庁は雪や路面の凍結による交通への影響、
[00:00:05.179 --> 00:00:12.620] 暴風雪や高波に警戒するとともに雪崩や屋根からの落雪にも十分注意するよう呼びかけています。
```

## Requirements

This model requires additional module.
```
pip3 install librosa
pip3 install sentencepiece
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 reazon_speech2.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 reazon_speech2.py --input AUDIO_FILE
```

## Reference

- [ReazonSpeech](https://research.reazon.jp/projects/ReazonSpeech/)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[reazonspeech-nemo-v2_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/reazon_speech2/reazonspeech-nemo-v2_encoder.onnx.prototxt)  
[reazonspeech-nemo-v2_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/reazon_speech2/reazonspeech-nemo-v2_decoder.onnx.prototxt)  
[reazonspeech-nemo-v2_joint.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/reazon_speech2/reazonspeech-nemo-v2_joint.onnx.prototxt)  
