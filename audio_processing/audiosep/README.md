# AudioSep: Separate Anything You Describe

## Input

Audio file

[Mixed audio file](./input.wav)
This audio data was adapted from the [official audiosep implementation](https://github.com/Audio-AGI/AudioSep)

Text condition

## Output

Audio file

- Estimated audio

https://user-images.githubusercontent.com/29946532/149924467-124a6605-1a52-41ce-8d79-bc54335a0f28.mov

- Ground truth

https://user-images.githubusercontent.com/29946532/149924495-1398c57b-5e8a-4012-8e9a-97e655a3ea26.mov

(Audio from http://swpark.me/voicefilter/)

## Usage
Internet connection is required when running the script for the first time, as the model files will be automatically downloaded.

Separate the sound described by the language query.
```bash
$ python3 audiosep.py -i thunder
```

<movie>
    <source src = "./res/output_thunder.mov" type="video/mov">
</movie>

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
