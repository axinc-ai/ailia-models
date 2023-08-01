# Retrieval-based-Voice-Conversion

## Input

Audio file

(Audio from https://github.com/ohashi3399/RVC-demo)

## Output

Audio file

## Requirements

This model requires additional module.
```
pip3 install librosa
pip3 install soundfile
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 rvc.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 rvc.py --input AUDIO_FILE
```

By adding the `--model_file` option, you can specify vc model file.
```bash
$ python3 rvc.py --model_file AISO-HOWATTO.onnx
```

## Reference

- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [RVC向け学習済みボイスモデルデータ](https://chihaya369.booth.pm/items/4701666)

## Framework

Pytorch

## Model Format

ONNX opset=14

## Netron

[hubert_base.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rvc/hubert_base.onnx.prototxt)  
[AISO-HOWATTO.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rvc/AISO-HOWATTO.onnx.prototxt)
