# EasyOCR
Ready-to-use OCR with 80+ supported languages and all popular writing scripts including Latin, Chinese, Arabic, Devanagari, Cyrillic and etc.

## Input

![input image](./example/chinese.jpg)

(from https://github.com/JaidedAI/EasyOCR/tree/master/examples)

## Output

recognized text    
shape is (binding box, recognized text, confidence score)

```
recognize result
([[86, 80], [134, 80], [134, 128], [86, 128]], '西', 0.5986109503715511)
([[187, 75], [469, 75], [469, 165], [187, 165]], '愚园路', 0.7509350273962178)
([[517, 81], [565, 81], [565, 123], [517, 123]], '东', 0.9941764272051365)
([[78, 126], [136, 126], [136, 156], [78, 156]], '315', 0.9999935304043551)
([[514, 124], [574, 124], [574, 156], [514, 156]], '309', 0.9999765305361424)
([[81, 175], [125, 175], [125, 211], [81, 211]], 'I', 0.9425757451290373)
([[226, 171], [414, 171], [414, 220], [226, 220]], 'Yyan Rd', 0.6217804142448365)
([[529, 173], [569, 173], [569, 213], [529, 213]], 'E', 0.03826956319031538)
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 easyocr.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 easyocr.py --input IMAGE_PATH
```

By adding the `--language` option, you can choose the language.
```bash
$ python3 easyocr.py --language LANGUAGE
$ python3 easyocr.py -l LANGUAGE
(ex) $ python3 easyocr.py --language japanese
(ex) $ python3 easyocr.py -l japanese
```

The prepared language are as follows.
  - chinese (default)
  - japanese
  - english
  - french
  - korean
  - thai

## Reference
[Jaided AI: EasyOCR demo](https://www.jaided.ai/easyocr/)    
[Ready-to-use OCR with 80+ supported languages and all popular writing scripts including Latin, Chinese, Arabic, Devanagari, Cyrillic and etc.](https://github.com/JaidedAI/EasyOCR)

## Framework
Pytorch    

## Model Format
ONNX opset = 11    

## Netron
[detector_craft.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/easyocr/detector_craft.onnx.prototxt)   
[recognizer_zh_sim_g2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/easyocr/recognizer_zh_sim_g2.onnx.prototxt)    
[recognizer_japanese_g2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/easyocr/recognizer_japanese_g2.onnx.prototxt)    
[recognizer_english_g2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/easyocr/recognizer_english_g2.onnx.prototxt)    
[recognizer_latin_g2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/easyocr/recognizer_latin_g2.onnx.prototxt)    
[recognizer_korean_g2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/easyocr/recognizer_korean_g2.onnx.prototxt)    
[recognizer_thai.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/easyocr/recognizer_thai.onnx.prototxt)      
