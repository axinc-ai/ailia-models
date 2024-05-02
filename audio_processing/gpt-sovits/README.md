# GPT-SoVITS

### Input
- A audio and transcript for voice cloning

### Output
The Voice file is output as .wav which path is defined as `SAVE_WAV_PATH` in `gpt-sovits.py `.  

### Requirements
This model requires pyopenjtalk for g2p.

```
pip3 install -r requirements.txt
```

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample sentence,
```
python3 gpt-sovits.py 
```

Run with audio prompt.

```
python3 gpt-sovits.py -i "音声合成のテストを行なっています。" --audio reference_audio_captured_by_ax.wav --transcript "水をマレーシアから買わなくてはならないのです" -e 1
```

### Reference
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

### Framework
PyTorch 2.1.2

### Model Format
ONNX opset = 17

### Netron

- [cnhubert.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/cnhubert.onnx.prototxt)
- [t2s_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/t2s_encoder.onnx.prototxt)
- [t2s_fsdec.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/t2s_fsdec.onnx.prototxt)
- [t2s_sdec.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/t2s_sdec.onnx.prototxt)
- [vits.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/vits.onnx.prototxt)
