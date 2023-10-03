# NDL OCR

## Input

![Input](demo.png)

(Image from [東洋学芸雑誌 第1号](https://dglb01.ninjal.ac.jp/ninjaldl/bunken.php?title=toyogakuge))

## Output

- Recognized text
```bash
recognized: 號壹第誌雜勢學洋東
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 ndlocr_text_recognition.py
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 ndlocr_text_recognition.py --input IMAGE_PATH
```

It automatically recognizes whether the text is written vertically or horizontally, but you can also specify that it is written vertically with the `--vert` option.
```bash
$ python3 ndlocr_text_recognition.py --vert
```

## Reference

- [text_recognition](https://github.com/ndl-lab/text_recognition)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[ndlenfixed64-mj0-synth1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ndlocr_text_recognition/ndlenfixed64-mj0-synth1.onnx.prototxt)
