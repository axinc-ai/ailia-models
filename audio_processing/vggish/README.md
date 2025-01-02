# vggish : a feature embedding frontend for audio classification

## Input

Audio file
```
Wav file from Public Domain https://soundbible.com/1698-Public-Transit-Bus.html
Default input: bus_chatter.wav
```


## Output

feature (numpy file)
```
feature.npy
```


## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 vggish.py
```

If you want to specify the input file, put the path after the `--input` option.
```bash
$ python3 vggish.py --input AUDIO_PATH
```
```


## Reference

[VGGish](https://github.com/harritaylor/torchvggish)

## Framework

Pytorch

## Model Format

ONNX opset=14

## Netron

[vggish.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vggish/vggish.onnx.prototxt)  