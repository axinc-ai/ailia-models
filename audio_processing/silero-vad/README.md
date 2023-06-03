# Silero VAD

## Input

Audio file
(https://models.silero.ai/vad_models/en.wav)

## Output

Audio file with silence removed

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 sileo-vad.py --input ex_example.wav --output only_speech.wav
```

## Reference

- [Silero VAD](https://github.com/snakers4/silero-vad)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[silero_vad.onnx](https://netron.app/?url=https://storage.googleapis.com/ailia-models/silero-vad/silero_vad.onnx)  
