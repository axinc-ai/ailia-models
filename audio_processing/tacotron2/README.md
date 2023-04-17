# Tacotron2

### Input
A sentence which is defined as `SENTENCE` in `tacotron2.py`.  

### Output
The Voice file is output as .wav which path is defined as `SAVE_WAV_PATH` in `tacotron2.py`.  

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.


`SENTENCE` is defined in the `tacotron2.py`.
ex. SENTENCE = 'The boy was there when the sun rose.'

For the sample sentence,
```
python3 tacotron2.py 
```

If you want to specify the input sentence, put the wav path after the --input option.
You can use --savepath option to change the name of the output file to save.

```
python3 tacotron2.py  --input SENTENCE --savepath SAVE_WAV_PATH
```

### For Japanese

Recognizing Japanese requires converting the text into phonemes. Conversion to phonemes requires openjtalk.

```
pip3 install pyopenjtalk
```

### Reference
[Tacotron2](https://github.com/NVIDIA/tacotron2)  
[ONNX Export](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2/tensorrt)

### Framework
PyTorch

### Model Format
ONNX opset = 11, 12

### Netron

- [decoder_iter.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/tacotron2/decoder_iter.onnx.prototxt)
- [encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/tacotron2/encoder.onnx.prototxt)
- [postnet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/tacotron2/postnet.onnx.prototxt)
- [waveglow.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/tacotron2/waveglow.onnx.prototxt)
