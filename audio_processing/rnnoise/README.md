# rnnoise

## Input

Audio file

- Sample rate: 48 kHz
- Bit per sample: 16-bit
- Bit rate: 768 kbps

###

(Audio from https://jmvalin.ca/demo/rnnoise/)

## Output

Audio file

###

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 rnnoise.py
```

If you want to specify the audio, put the file path after the `--input` option.
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 rnnoise.py --input AUDIO_FILE --savepath SAVE_AUDIO_FILE
```

## Reference

- [rnnoise](https://github.com/xiph/rnnoise)
- [xiph.org / moz://a](https://jmvalin.ca/demo/rnnoise/)

## Framework

Keras

## Model Format

ONNX opset=14

## Netron

[rnn_model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rnnoise/rnn_model.onnx.prototxt)
