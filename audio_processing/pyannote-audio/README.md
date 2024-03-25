# Pyannote-audio : Speaker Diarization

## Input

Audio file (.wav format).
```
Example
input: data/demo.wav
```
(Wav file from https://github.com/pyannote/pyannote-audio/tree/develop/pyannote/audio/sample)

## Output

When and who spoke.
```
[ 00:00:06.714 -->  00:00:07.003] A SPEAKER_02
[ 00:00:07.003 -->  00:00:07.173] B SPEAKER_00
[ 00:00:07.580 -->  00:00:07.597] C SPEAKER_00
[ 00:00:07.597 -->  00:00:08.276] D SPEAKER_02
[ 00:00:08.276 -->  00:00:08.293] E SPEAKER_00
[ 00:00:08.293 -->  00:00:08.310] F SPEAKER_02
[ 00:00:08.310 -->  00:00:09.906] G SPEAKER_00
[ 00:00:09.906 -->  00:00:10.959] H SPEAKER_02
[ 00:00:10.466 -->  00:00:14.745] I SPEAKER_00
[ 00:00:10.959 -->  00:00:10.976] J SPEAKER_01
[ 00:00:14.303 -->  00:00:17.886] K SPEAKER_01
[ 00:00:18.022 -->  00:00:21.502] L SPEAKER_00
[ 00:00:18.157 -->  00:00:18.446] M SPEAKER_01
[ 00:00:21.774 -->  00:00:28.531] N SPEAKER_01
[ 00:00:27.886 -->  00:00:29.991] O SPEAKER_00
```

## Requirements

This model recommends additional module.
```bash
$ pip3 install -r requirements.txt
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample
```bash
$ python pyannote-audio.py
```

If you want to specify the audio, put the file path after the `--insu` pr `-input_sample` option.

```bash
$ python pyannote-audio.py --insu FILE_PATH
```

If you know the number of speakers, put the numper　`--num` or `-num_speaker` option.
```bash
$ python pyannote-audio.py --num 2
```

If you know the maxisimum number of speakers, put the numper　`--max` or `-max_speaker` option.
```bash
$ python pyannote-audio.py --max 4
```

If you know the minimum number of speakers, put the numper　`--min` or `-min_speaker` option.
```bash
$ python pyannote-audio.py --min 2
```

By giving the `--use_onnx` option, you can use onnx.
```bash
$ python pyannote-audio.py --use_onnx
```

By giving the `--embed` option, you can get embedding vector in the input file.
```bash
$ python pyannote-audio.py --embed
```

## Reference

- [Pyannte-audio](https://github.com/pyannote/pyannote-audio)
- [Hugging Face - pyannote in speaker-deriazation](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [Hugging Face - hdbraib in wespeaker-voxceleb-resnet34-LM](https://huggingface.co/hbredin/wespeaker-voxceleb-resnet34-LM/tree/main)
- [KaldiFeat](https://github.com/yuyq96/kaldifeat)

## Framework

Onnxruntime

## Model Format

ONNX opset=14,17

## Netron
***
作成中
***