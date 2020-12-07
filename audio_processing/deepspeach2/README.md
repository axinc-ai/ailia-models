# deepspeach2

### input

audio file

```
LibriSpeech ASR corpus
http://www.openslr.org/12
1221-135766-0000.wav
```

### output

texts

```
#librispeech_pretrained_v2
how strange it seemed to the sad woman as she watched the growth and the beauty that became every day more brilliant and the intelligence that through its quivering sunshine over the tiny features of this child

#an4_pretrained_v2
sthiee sixtysx s one cs one stwp teoh ten teny kwenth three t four eineaieteen twonr two seven te ine  thine shirn i np twe tseiox sven sie

#ted_pretrained_v2
howstrange at seemed to the sad woman she wachd the grolt han the beauty that became every day more brilliant and the intelligence that through its equivering sunching over the tiny peacturs of this child
```

### Usage

```bash
$ python3 deepspeech2.py -i 1221-135766-0000.wav
```

### Reference
[deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)  

### Framework
PyTorch

### Model Format
ONNX opset = 10

### Netron

[an4_pretrained_v2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deepspeach2/an4_pretrained_v2.onnx.prototxt)
[librispeech_pretrained_v2.onnx.prototxt(https://netron.app/?url=https://storage.googleapis.com/ailia-models/deepspeach2/librispeech_pretrained_v2.onnx.prototxt)
[ted_pretrained_v2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deepspeach2/ted_pretrained_v2.onnx.prototxt)
