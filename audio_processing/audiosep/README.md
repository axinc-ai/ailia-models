# AudioSep: Separate Anything You Describe

## Input

**Mixed audio file**
Audio file in wav format with mixed sources. [input.wav](./input.wav)
This audio file was adapted from the [official audiosep implementation](https://github.com/Audio-AGI/AudioSep)

**Text condition**
Text description of the sound source you want to separate.

## Output

**Audio file**
* Separated audio according to the text query

## Usage
Internet connection is required when running the script for the first time, as the model files will be automatically downloaded.

Separate the sound described by the language query.

#### Example1: Extract sound of thunder
```bash
$ python3 audiosep.py -i "thunder" -p input.wav -s output_thunder.wav
```


#### Example2: Extract sound of waterdrops
```bash
$ python3 audiosep.py -i "water drops" -p input.wav -s output_waterdrops.wav
```

```.wav``` file containing the sound source separated from the original mixture will be created in both cases.

## Reference

* [AudioSep](https://github.com/Audio-AGI/AudioSep)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[text_encoder.onnx.prototxt]()
[resunet.onnx.prototxt]()