# g2pE: A Simple Python Module for English Grapheme To Phoneme Conversion

## Input

```
I'm an activationist.
```

```
To be or not to be, that is the question
```

## Output

```
['AY1', 'M', ' ', 'AE1', 'N', ' ', 'AE2', 'K', 'T', 'IH0', 'V', 'EY1', 'SH', 'AH0', 'N', 'IH0', 'S', 'T', ' ', '.']
```

```
['T', 'UW1', ' ', 'B', 'IY1', ' ', 'AO1', 'R', ' ', 'N', 'AA1', 'T', ' ', 'T', 'UW1', ' ', 'B', 'IY1', ' ', ',', ' ', 'DH', 'AE1', 'T', ' ', 'IH1', 'Z', ' ', 'DH', 'AH0', ' ', 'K', 'W', 'EH1', 'S', 'CH', 'AH0', 'N']
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 g2p_en.py
```

If you want to specify the input text, put the image path after the `--input` option.
```bash
$ python3 g2p_en.py --input text
```

## Reference

- [g2p_en](https://github.com/Kyubyong/g2p)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[g2p_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/g2p_en/g2p_encoder.onnx.prototxt)
[g2p_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/g2p_en/g2p_decoder.onnx.prototxt)
