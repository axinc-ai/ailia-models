# Zonos

### Input
A synthetic text and reference audio for audio duplication

### Output
The Voice file is output as .wav which path is defined as `SAVE_WAV_PATH` in `zonos.py `.

### Requirements
This model requires additional module.

```
pip3 install -r requirements.txt
```

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample sentence and sample audio,
```
python3 zonos.py 
```

Run with audio prompt.

```
python3 zonos.py -i "こんにちは" --ref_audio exampleaudio.mp3
```

Run for english.

```
python3 zonos.py -i "Hello, world!" --text_language en-us --ref_audio exampleaudio.mp3
```

### Reference
[Zonos](https://github.com/Zyphra/Zonos)

### Framework
PyTorch

### Model Format
ONNX opset = 17

### Netron

#### Normal model

- [speaker_embedding.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/zonos/speaker_embedding.onnx.prototxt)
- [phoneme_embedder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/zonos/phoneme_embedder.onnx.prototxt)
- [conditioner.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/zonos/conditioner.onnx.prototxt)
- [generator_first.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/zonos/generator_first.onnx.prototxt)
- [generator_stage.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/zonos/generator_stage.onnx.prototxt)
- [autoencoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/zonos/autoencoder.onnx.prototxt)

