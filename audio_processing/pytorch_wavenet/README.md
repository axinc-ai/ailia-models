# pytorch_wavenet

### Input

audio file

- For the sample: first_sample.wav

### Output

audio file

## Requirements

This model requires additional module.

```
pip3 install soundfile
```

### Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 pytorch_wavenet.py
```

If you want to specify the input audio file, put the wav path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 pytorch_wavenet.py --input WAV_PATH --savepath SAVE_WAV_PATH
```

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

ONNX opset = 17

### Netron

[wavenet_pytorch_op_17.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/pytorch_wavenet/wavenet_pytorch_op_17.onnx.prototxt)
