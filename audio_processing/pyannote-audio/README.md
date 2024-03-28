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
![Output](output.png)

```
[ 00:00:06.714 -->  00:00:07.003] A speaker91
[ 00:00:07.003 -->  00:00:07.173] B speaker90
[ 00:00:07.580 -->  00:00:08.310] C speaker91
[ 00:00:08.310 -->  00:00:09.923] D speaker90
[ 00:00:09.923 -->  00:00:10.976] E speaker91
[ 00:00:10.466 -->  00:00:14.745] F speaker90
[ 00:00:14.303 -->  00:00:17.886] G speaker91
[ 00:00:18.022 -->  00:00:21.502] H speaker90
[ 00:00:18.157 -->  00:00:18.446] I speaker91
[ 00:00:21.774 -->  00:00:28.531] J speaker91
[ 00:00:27.886 -->  00:00:29.991] K speaker90
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

If you want to specify the audio, put the file path after the `--i` or `-input` option.

```bash
$ python pyannote-audio.py --i FILE_PATH
```

If you want to specify the ground truth, put the file path after the `--ig` or `-input_ground` option.

```bash
$ python pyannote-audio.py --ig FILE_PATH
```

If you want to specify the output file, put the file path after the `--o` or `-output` option.

```bash
$ python pyannote-audio.py --o FILE_PATH
```

If you want to specify the output ground truth file, put the file path after the `--og` or `-output_ground` option.

```bash
$ python pyannote-audio.py --og FILE_PATH
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

By giving the `--e` or `-error` option, you can get diarization error rate.
```bash
$ python pyannote-audio.py --use_onnx
```

By giving the `--plt` option, you can visualize results.
```bash
$ python pyannote-audio.py --use_onnx
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
- [Hugging Face - pyannote in speaker-diariazation](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [Hugging Face - hdbrain in wespeaker-voxceleb-resnet34-LM](https://huggingface.co/hbredin/wespeaker-voxceleb-resnet34-LM/tree/main)
- [KaldiFeat](https://github.com/yuyq96/kaldifeat)

## Framework

Onnxruntime

## Model Format

ONNX opset=14,17

## Netron

[segmentation.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/pyannote-audio/segmentation.onnx.prototxt)
[speaker-embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/pyannote-audio/speaker-embedding.onnx.prototxt)
