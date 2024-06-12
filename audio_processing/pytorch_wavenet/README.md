# pytorch_wavenet

### Input

audio file (randomly generated)

### Output

audio file

### Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

In this version, you have to use `--onnx` option.

```bash
$ python3 pytorch_wavenet.py --onnx
```

After running this program, output.wav, which is predicted besed on randomly generated wave data is generated.

### Reference

[pytorch_wavenet](https://github.com/vincentherrmann/pytorch-wavenet)  

### Framework

PyTorch 1.5.0

### Model Format

ONNX opset = 11

### Netron

[pytorch_wavenet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/)
