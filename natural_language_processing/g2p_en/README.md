# G2P EN

## Input

```
texts = ["I have $250 in my pocket.", # number -> spell-out
          "popular pets, e.g. cats and dogs", # e.g. -> for example
          "I refuse to collect the refuse around here.", # homograph
          "I'm an activationist."] # newly coined word
```

## Output

```
['AY1', ' ', 'HH', 'AE1', 'V', ' ', 'T', 'UW1', ' ', 'HH', 'AH1', 'N', 'D', 'R', 'AH0', 'D', ' ', 'F', 'IH1', 'F', 'T', 'IY0', ' ', 'D', 'AA1', 'L', 'ER0', 'Z', ' ', 'IH0', 'N', ' ', 'M', 'AY1', ' ', 'P', 'AA1', 'K', 'AH0', 'T', ' ', '.']
['P', 'AA1', 'P', 'Y', 'AH0', 'L', 'ER0', ' ', 'P', 'EH1', 'T', 'S', ' ', ',', ' ', 'F', 'AO1', 'R', ' ', 'IH0', 'G', 'Z', 'AE1', 'M', 'P', 'AH0', 'L', ' ', 'K', 'AE1', 'T', 'S', ' ', 'AH0', 'N', 'D', ' ', 'D', 'AA1', 'G', 'Z']
['AY1', ' ', 'R', 'IH0', 'F', 'Y', 'UW1', 'Z', ' ', 'T', 'UW1', ' ', 'K', 'AH0', 'L', 'EH1', 'K', 'T', ' ', 'DH', 'AH0', ' ', 'R', 'EH1', 'F', 'Y', 'UW2', 'Z', ' ', 'ER0', 'AW1', 'N', 'D', ' ', 'HH', 'IY1', 'R', ' ', '.']
['AY1', 'M', ' ', 'AE1', 'N', ' ', 'AE2', 'K', 'T', 'IH0', 'V', 'EY1', 'SH', 'AH0', 'N', 'IH0', 'S', 'T', ' ', '.']
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
