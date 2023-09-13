# VALL-E X: Multilingual Text-to-Speech Synthesis and Voice Cloning.

### Input
- A sentence for text to speech
- A audio and transcript for voice cloning

### Output
The Voice file is output as .wav which path is defined as `SAVE_WAV_PATH` in `vall-e-x.py`.  

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample sentence,
```
python3 vall-e-x.py 
```

If you want to specify the input sentence, put the wav path after the --input option.
You can use --savepath option to change the name of the output file to save.

```
python3 vall-e-x.py --input "Hello world." --savepath SAVE_WAV_PATH
```

### For Japanese

Recognizing Japanese requires converting the text into phonemes. Conversion to phonemes requires openjtalk.

```
# for macOS, Linux
pip3 install pyopenjtalk
# for Windows
pip3 install pyopenjtalk-prebuilt
```

Run.

```
python3 vall-e-x.py -i "こんにちは。"
```

### Reference
[VALL-E-X](https://github.com/Plachtaa/VALL-E-X)

### Framework
PyTorch

### Model Format
ONNX opset = 15

### Netron

- [ar_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/ar_decoder.onnx.prototxt)
