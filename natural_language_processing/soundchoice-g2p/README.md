# SoundChoice: Grapheme-to-Phoneme Models with Semantic Disambiguation

## Input

Text to recognize

- Example
```
To be or not to be, that is the question
```

## Output

Phoneme
```
T-UW- -B-IY- -AO-R- -N-AA-T- -T-UW- -B-IY- -DH-AE-T- -IH-Z- -DH-AH- -K-W-EH-S-CH-AH-N
```

## Requirements
This model requires additional module.

```
pip3 install transformers
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample text,
```bash
$ python3 soundchoice-g2p.py
```

If you want to specify the input text, put the text after the `--input` option.
```bash
$ python3 soundchoice-g2p.py --input TEXT
```

## Reference

- [Hugging Face - speechbrain/soundchoice-g2p](https://huggingface.co/speechbrain/soundchoice-g2p)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[soundchoice-g2p_atn.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/soundchoice-g2p/soundchoice-g2p_atn.onnx.prototxt)
[soundchoice-g2p_emb.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/soundchoice-g2p/soundchoice-g2p_emb.onnx.prototxt)
[rnn_beam_searcher.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/soundchoice-g2p/rnn_beam_searcher.onnx.prototxt)
