# Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention

### Input
A sentence which is defined as `SENTENCE` in `pytorch-dc-tts.py`.  

### Output
The Voice file is output as .wav which path is defined as `SAVE_WAV_PATH` in `pytorch-dc-tts.py`.  

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.


`SENTENCE` is defined in the `pytorch-dc-tts.py`.
ex. SENTENCE = 'The boy was there when the sun rose.'

For the sample sentence,
```
python3 pytorch-dc-tts.py 
```

If you want to specify the input sentence, put the wav path after the --input option.
You can use --savepath option to change the name of the output file to save.

```
python3 pytorch-dc-tts.py  --input SENTENCE --savepath SAVE_WAV_PATH
```


### Reference
[Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://github.com/tugstugi/pytorch-dc-tts)  

### Framework
PyTorch

### Model Format
ONNX opset = 10

### Netron

- [text2mel.onnx.prototxt]()
- [ssrn.onnx.prototxt]()
