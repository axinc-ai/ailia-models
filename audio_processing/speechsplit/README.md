# Unsupervised Speech Decomposition Via Triple Information Bottleneck

## Input

Audio file (.wav file)


## Output

Converted audio file (.wav file)


## Requirements

This model requires additional module.
```
```

This model uses wavenet_vocoder as external model. If you use wavenet_vocoder as it is original, you have to install additional module as below and download checkpoint_step001000000_ema.pth from https://github.com/auspicious3000/autovc. See detail in https://github.com/r9y9/wavenet_vocoder.
```
pip3 install wavenet_vocoder==0.1.1
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 speechsplit.py
```

## Reference

- [SpeechSplit](https://github.com/auspicious3000/SpeechSplit)

- [WaveNet vocoder](https://github.com/r9y9/wavenet_vocoder)

- [autovc](https://github.com/auspicious3000/autovc)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron