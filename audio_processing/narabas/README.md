# narabas: Japanese phoneme forced alignment tool

## Input

Wav file(sample file: `input.wav`)


## Output

Assigned phoneme and corresponding start and end times. Output for sample input wav file is below.

```
INFO narabas.py (192) : 0.220 0.300 a
INFO narabas.py (192) : 0.300 0.320 n
INFO narabas.py (192) : 0.320 0.400 e
INFO narabas.py (192) : 0.400 0.440 m
INFO narabas.py (192) : 0.440 0.480 u
INFO narabas.py (192) : 0.480 0.520 s
INFO narabas.py (192) : 0.520 0.580 u
INFO narabas.py (192) : 0.580 0.620 m
INFO narabas.py (192) : 0.620 0.680 e
INFO narabas.py (192) : 0.680 0.720 n
INFO narabas.py (192) : 0.720 0.860 o
INFO narabas.py (192) : 0.860 0.880 ts
INFO narabas.py (192) : 0.880 0.980 u
INFO narabas.py (192) : 0.980 1.060 i
INFO narabas.py (192) : 1.060 1.100 k
INFO narabas.py (192) : 1.100 1.160 o
INFO narabas.py (192) : 1.160 1.220 w
INFO narabas.py (192) : 1.220 1.620 a
```

## Usage

For the sample input wav file,
```bash
python3 narabas.py
```

If you want to infer with onnxruntime,
```bash
python3 narabas.py --onnx
```

To specify input wav file,
```bash
python3 narabas.py --input <input wav file>
```

## Reference
[narabas](https://github.com/darashi/narabas)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron
[narabas-v0.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/audio_processing/narabas/narabas-vo.onnx.prototxt)
