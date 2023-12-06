# AudioSep: Separate Anything You Describe

## Input

* **Mixed audio file**

Audio file in wav format with mixed sources. [input.wav](./input.wav)

https://github.com/axinc-ai/ailia-models/assets/53651931/4b761212-a1c7-46dc-b598-a08e4c5ab7ff

This audio file was adapted from the [official audiosep implementation](https://github.com/Audio-AGI/AudioSep)

* **Text condition**

Text description of the sound source you want to separate.

## Output

* **Audio file**

Separated audio source according to the text query.

Saves to ```./output.wav``` by default but it can be specified with the ```--path``` option 

## Usage
Internet connection is required when running the script for the first time, as the model files will be automatically downloaded.

Running this script will separate sound sources from the original input audio file, according to the language query.

#### Example1: Extract sound of thunder
```bash
$ python3 audiosep.py -i "thunder" -p input.wav -s output_thunder.wav
```
https://github.com/axinc-ai/ailia-models/assets/53651931/d0d016dd-a808-4eb6-a4b5-9791f8f1bd2f

#### Example2: Extract sound of waterdrops
```bash
$ python3 audiosep.py -i "water drops" -p input.wav -s output_waterdrops.wav
```
https://github.com/axinc-ai/ailia-models/assets/53651931/7710b6c9-49dc-4d2a-8489-ccbf7fb45591

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



