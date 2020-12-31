# deepspeech2

### input

audio file（16kHz)

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

#### Basic

File input

```bash
$ python3 deepspeech2.py -i 1221-135766-0000.wav -s output.txt
```

Mic input

```bash
$ python3 deepspeech2.py -V
```

1. speak into the microphone when "Please speak something."
2. end the recording after about 1 second of silence and do voice recognition
3. return to 1 again after displaying the forecast results
4. type ``Ctrl+c`` if you want to exit

#### Options

With the `-d` option, decode the recognition results in BeamDecoder using the language model. With the `-a` option, you can use other trained models.

### Setup

#### Install pyaudio

Mac OS:
```
brew install portaudio
pip install pyaudio
```

Linux:
```
sudo apt-get install portaudio19-dev
pip install pyaudio
```

#### Install ctcdecode (for -d option)
This module is required to perform Beam Decode using the language model.

```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

### Reference
[deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)  

### Framework
PyTorch

### Model Format
ONNX opset = 10

### Netron

[an4_pretrained_v2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deepspeech2/an4_pretrained_v2.onnx.prototxt)

[librispeech_pretrained_v2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deepspeech2/librispeech_pretrained_v2.onnx.prototxt)

[ted_pretrained_v2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deepspeech2/ted_pretrained_v2.onnx.prototxt)
