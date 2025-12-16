# SenseVoice

## Input

Audio file

https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/main/example

## Output

Recognized speech text
```
うちの中学は弁当制で持っていきない場合は50円の学校販売のパンを買う
```

## Requirements

This model requires additional module.
```
pip3 install kaldi_native_fbank, sentencepiece
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 sensevoice.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 sensevoice.py --input AUDIO_FILE
```

## Reference

- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)

## Framework

Pytorch

## Model Format

ONNX opset=14

## Netron

[sensevoice_small.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sensevoice/sensevoice_small.onnx.prototxt)  
