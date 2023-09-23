# Dual-signal Transformation LSTM Network

## input

audio file（16kHz)

```
LibriSpeech ASR corpus
http://www.openslr.org/12
1221-135766-0000.wav
```

## Output

Audio file with noise removed

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 dtln.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 dtln.py --input AUDIO_FILE
```
If you run by onnxruntime instead of ailia, you use --onnx option.



## Reference

- [Dual-signal Transformation LSTM Network](https://github.com/breizhn/DTLN)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

- [dtln1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/dtln/dtln1.onnx.prototxt)  
- [dtln2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/dtln2/dtln2.onnx.prototxt)