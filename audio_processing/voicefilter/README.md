# VoiceFilter

## Input

Audio file

- Mixed audio

https://user-images.githubusercontent.com/29946532/149924386-1a67ec3c-390a-422e-b0b3-9540dff58c72.mov

- Reference audio for d-vector

https://user-images.githubusercontent.com/29946532/149924422-3620f0fb-dca7-45a9-a465-c22bfb00d7ae.mov

Input an audio file that is spoken by multiple people and an audio file that contains the voices of the people you want to extract.
The voice of one person is extracted and output.

## Output

Audio file

- Estimated audio

https://user-images.githubusercontent.com/29946532/149924467-124a6605-1a52-41ce-8d79-bc54335a0f28.mov

- Ground truth

https://user-images.githubusercontent.com/29946532/149924495-1398c57b-5e8a-4012-8e9a-97e655a3ea26.mov

(Audio from http://swpark.me/voicefilter/)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 voicefilter.py --input mixed.wav --reference_file ref-voice.wav
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
