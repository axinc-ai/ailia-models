# VoiceFilter

## Input

Audio file

- Mixed audio

https://user-images.githubusercontent.com/29946532/149611959-9a055cf4-fe38-4204-8e0d-f40630de35c6.mov

- Reference audio for d-vector

https://user-images.githubusercontent.com/29946532/149612070-8905cebc-24de-4b55-b903-04bb857f645c.mov

## Output

Audio file

- Estimated audio

https://user-images.githubusercontent.com/29946532/149612149-9517ec76-821a-4a88-a9d3-01cfdf82dc77.mov

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 voicefilter.py --input 000006-mixed.wav --reference_file 000006-target.wav
```

If you want to specify the mixed audio, put the file path after the `--input` option, and to specify the reference audio, put the file path after the `--reference_file` option.   
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