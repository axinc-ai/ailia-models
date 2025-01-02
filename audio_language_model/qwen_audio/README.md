# Qwen-Audio

## Input

- Audio file

  https://github.com/QwenLM/Qwen-Audio/blob/main/assets/audio/1272-128104-0000.flac

- Prompt

  what does the person say?

## Output

The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".

## Requirements

This model requires additional module.
```
pip3 install transformers
pip3 install tiktoken
pip3 install librosa
```


## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 qwen_audio.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 qwen_audio.py --input AUDIO_FILE
```

If you want to specify the prompt, put the prompt after the `--prompt` option.  
```bash
$ python3 qwen_audio.py --prompt PROMPT
```

## Reference

- [Qwen-Audio](https://github.com/QwenLM/Qwen-Audio)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[Qwen-Audio-Chat_encode.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/qwen_audio/Qwen-Audio-Chat_encode.onnx.prototxt)  
[Qwen-Audio-Chat.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/qwen_audio/Qwen-Audio-Chat.onnx.prototxt)  
