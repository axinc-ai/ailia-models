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

- [nar_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_decoder.onnx.prototxt)
- [nar_predict_layers.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_predict_layers.onnx.prototxt)
- [ar_audio_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/ar_audio_embedding.onnx.prototxt)
- [ar_language_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/ar_language_embedding.onnx.prototxt)
- [ar_text_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/ar_text_embedding.onnx.prototxt)
- [nar_audio_embeddings_0.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_audio_embeddings_0.onnx.prototxt)
- [nar_audio_embeddings_1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_audio_embeddings_1.onnx.prototxt)
- [nar_audio_embeddings_2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_audio_embeddings_2.onnx.prototxt)
- [nar_audio_embeddings_3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_audio_embeddings_3.onnx.prototxt)
- [nar_audio_embeddings_4.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_audio_embeddings_4.onnx.prototxt)
- [nar_audio_embeddings_5.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_audio_embeddings_5.onnx.prototxt)
- [nar_audio_embeddings_6.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_audio_embeddings_6.onnx.prototxt)
- [nar_audio_embeddings_7.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_audio_embeddings_7.onnx.prototxt)
- [nar_language_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_language_embedding.onnx.prototxt)
- [nar_text_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_text_embedding.onnx.prototxt)
- [ar_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/ar_decoder.onnx.prototxt)
- [ar_decoder.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/ar_decoder.opt.onnx.prototxt)
- [encodec.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/encodec.onnx.prototxt)
- [vocos.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/vocos.onnx.prototxt)
- [audio_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/audio_embedding.onnx.prototxt)
