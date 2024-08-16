# VALL-E X: Multilingual Text-to-Speech Synthesis and Voice Cloning.

### Input
- A sentence for text to speech
- A audio and transcript for voice cloning

### Output
The Voice file is output as .wav which path is defined as `SAVE_WAV_PATH` in `vall-e-x.py`.  

### Requirements
This model requires pyopenjtalk for g2p.

```
pip3 install -r requirements.txt
```

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

Run with audio prompt.

```
python3 vall-e-x.py -i "音声合成のテストを行なっています。" --audio BASIC5000_0001.wav --transcript "水をマレーシアから買わなくてはならないのです" -e 1
```

### Reference
[VALL-E-X](https://github.com/Plachtaa/VALL-E-X)

### Framework
PyTorch 2.2.0.dev20230910

### Model Format
ONNX opset = 15

### Netron

- [nar_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_decoder.onnx.prototxt)
- [nar_predict_layers.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_predict_layers.onnx.prototxt)
- [ar_audio_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/ar_audio_embedding.onnx.prototxt)
- [ar_language_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/ar_language_embedding.onnx.prototxt)
- [ar_text_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/ar_text_embedding.onnx.prototxt)
- [nar_audio_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_audio_embedding.onnx.prototxt)
- [nar_audio_embedding_layers.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_audio_embedding_layers.onnx.prototxt)
- [nar_language_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_language_embedding.onnx.prototxt)
- [nar_text_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/nar_text_embedding.onnx.prototxt)
- [ar_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/ar_decoder.onnx.prototxt)
- [ar_decoder.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/ar_decoder.opt.onnx.prototxt)
- [encodec.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/encodec.onnx.prototxt)
- [vocos.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vall-e-x/vocos.onnx.prototxt)
