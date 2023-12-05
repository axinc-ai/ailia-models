# T5 Whisper Medical Error Correction

## Input

[TEXT file from whisper medium.](input.txt)

```
こんにちは、先生。最近手足の経連があります。
こんにちは。手足の経連がある場合、心臓の問題が考えられます。
経連の発作がいつ頻繁に起こるかを教えていただけますか?
経連はほとんど毎日です。
```

## Output

Text after error correction.

```
こんにちは、先生。最近手足の痙攣があります。こんにちは。手足の痙攣がある場合、心臓の問題が考えられます。痙攣の発作がいつ頻繁に起こるかを教えていただけますか?痙攣はほとんど毎日です。
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample text,
```bash
$ python3 t5_whisper_medical.py
```

If you want to specify the text or pdf file, put the file path after the `-i` option.  
```bash
$ python3 t5_whisper_medical.py -i FILE_PATH
```

## Reference

- [onnxt5](https://github.com/abelriboulot/onnxt5)

### Framework
PyTorch

## Model Format

ONNX opset=12

## Netron

- [t5_whisper_medical-decoder-with-lm-head.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/t5_whisper_medical/t5_whisper_medical-decoder-with-lm-head.onnx.prototxt)
- [t5_whisper_medical-encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/t5_whisper_medical/t5_whisper_medical-encoder.onnx.prototxt)
