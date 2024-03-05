# Distil-Whisper

## Input

Audio file

https://user-images.githubusercontent.com/29946532/197575850-d4b76831-7a14-41f0-9253-8bcf477b3f7e.mov

## Output

Recognized speech text
```
He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.
```

## Requirements

This model requires additional module.
```
pip3 install transformers
pip3 install soundfile
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 distil-whisper.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 distil-whisper.py --input AUDIO_FILE
```

## Reference

- [Hugging Face - Distil-Whisper](https://github.com/huggingface/distil-whisper)
- [Whisper](https://github.com/openai/whisper)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[distil-large-v2_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/distil-whisper/distil-large-v2_encoder.onnx.prototxt)  
[distil-large-v2_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/distil-whisper/distil-large-v2_decoder.onnx.prototxt)  
