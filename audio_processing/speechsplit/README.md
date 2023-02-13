# Unsupervised Speech Decomposition Via Triple Information Bottleneck

## Input

Audio file (.wav file)

input_org.wav and input_trg.wav are from https://github.com/auspicious3000/SpeechSplit/blob/master/assets/demo.pkl


## Output

Converted audio file (.wav file)


## Requirements
This model requires pysptk for pre processing.

```
pip3 install pysptk
```

This model uses wavenet_vocoder as external model. If you use wavenet_vocoder as it is original, you have to install additional module as below and download `checkpoint_step001000000_ema.pth` from https://github.com/auspicious3000/autovc. See detail in https://github.com/r9y9/wavenet_vocoder.
```
pip3 install torch==1.12.0
pip3 install wavenet_vocoder==0.1.1
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 speechsplit.py
```

You can specify input audio files by adding `-i`, `-i2`, `-g`, `-g2` option. `-i` and `-i2` are path to input wav. `-g` and `-g2` are flag to distinguish whether the input speaker `-i` and `-i2` are male or female, respectively. You set `g=M` if the speaker is male and set `g=F` if the speaker is female.
```
python3 speechsplit.py -i [wav_file] -i2 [wav_file] -g M -g2 F
```

## Reference

[SpeechSplit](https://github.com/auspicious3000/SpeechSplit)

[WaveNet vocoder](https://github.com/r9y9/wavenet_vocoder)

[autovc](https://github.com/auspicious3000/autovc)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron
[F0_Converter.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/speechsplit/F0_Converter.prototxt)

[Generator.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/speechsplit/Generator.onnx.prototxt)