# Piano transcription

## Input

Piano recording (audio file)
```
cut_liszt.mp3
```

## Output

Piano recording transcribed to MIDI file
```
output.mid
```

## Requirements
This model requires additional module.

```
pip3 install audioread
pip3 install librosa
pip3 install mido
```

And requries additional package.
```
ffmpeg
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 piano_transcription.py
```

If you want to specify the audio file, put the file path after the `--input` option.   
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 piano_transcription.py --input AUDIO_FILE --savepath SAVE_PATH
```

## Reference

- [Piano transcription](https://github.com/bytedance/piano_transcription)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[note_model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/piano_transcription/note_model.onnx.prototxt)  
[pedal_model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/piano_transcription/pedal_model.onnx.prototxt)
