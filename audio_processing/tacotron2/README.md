# Tacotron2 : Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions.

### Input
A sentence for text to speech

### Output
The Voice file is output as .wav which path is defined as `SAVE_WAV_PATH` in `tacotron2.py`.  

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample sentence,
```
python3 tacotron2.py 
```

If you want to specify the input sentence, put the wav path after the --input option.
You can use --savepath option to change the name of the output file to save.

```
python3 tacotron2.py --input "Hello world." --savepath SAVE_WAV_PATH
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
python3 tacotron2.py -i "こんにちは。" -m tsukuyomi
```

### Reference
[Tacotron2](https://github.com/NVIDIA/tacotron2)  
[ONNX Export](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2/tensorrt)

### Framework
PyTorch

### Model Format
ONNX opset = 11, 12

### Netron

[NVIDIA Model](LICENSE)

- [decoder_iter.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/tacotron2/decoder_iter.onnx.prototxt)
- [encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/tacotron2/encoder.onnx.prototxt)
- [postnet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/tacotron2/postnet.onnx.prototxt)
- [waveglow.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/tacotron2/waveglow.onnx.prototxt)

[Tsukuyomi Chan Model](LICENSE_TSUKUYOMI)

- [tsukuyomi_accent_decoder_iter.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/tacotron2/tsukuyomi_accent_decoder_iter.onnx.prototxt)
- [tsukuyomi_accent_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/tacotron2/tsukuyomi_accent_encoder.onnx.prototxt)
- [tsukuyomi_accent_postnet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/tacotron2/tsukuyomi_accent_postnet.onnx.prototxt)
- [tsukuyomi_accent_waveglow.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/tacotron2/tsukuyomi_accent_waveglow.onnx.prototxt)
