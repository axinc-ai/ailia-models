# ReazonSpeech

## Input

Audio file

https://user-images.githubusercontent.com/29946532/218316186-266ffd00-4b1d-42c1-aa45-0080d781feb5.mov

## Output

Recognized speech text
```
気象庁は雪や路面の凍結による交通への影響暴風雪や高波に警戒するとともに雪崩や屋根からの落雪にも十分注意するよう呼びかけています
```

## Requirements

This model requires additional module.
```
pip3 install librosa
pip3 install six
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 reazon_speech.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 reazon_speech.py --input AUDIO_FILE
```

## Reference

- [ReazonSpeech](https://research.reazon.jp/projects/ReazonSpeech/)

## Framework

Pytorch

## Model Format

ONNX opset=17,opset=11

## Netron

[reazonspeech-espnet-v1-encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/reason_speech/reazonspeech-espnet-v1-encoder.onnx.prototxt)  
[reazonspeech-espnet-v1-decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/reason_speech/reazonspeech-espnet-v1-decoder.onnx.prototxt)  
[reazonspeech-espnet-v1-lm.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/reason_speech/reazonspeech-espnet-v1-lm.onnx.prototxt)  
[reazonspeech-espnet-v1-ctc.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/reason_speech/reazonspeech-espnet-v1-ctc.onnx.prototxt)
