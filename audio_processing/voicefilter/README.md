# VoiceFilter

## Input

Audio file
```
Mixed audio: Ex. 000006-mixed.wav
```
```
Reference audio for d-vector: Ex. 000006-target.wav
```

## Output

Audio file
```
Estimated audio: output.wav
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 voicefilter.py --input 000006-mixed.wav --reference_file 000006-target.wav
```

If you want to specify the mixed audio, put the file path after the `--input` option,  
and to specify the reference audio, put the file path after the `--reference_file` option.   
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 voicefilter.py --input MIXED_WAV --reference_file REFERENCE_WAV --savepath SAVE_PATH
```


## Reference

- [VoiceFilter](https://github.com/mindslab-ai/voicefilter)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[embedder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/voicefilter/embedder.onnx.prototxt)
[model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/voicefilter/model.onnx.prototxt)
