# Correct Japanese Sentence Using Bert

### input
A `SENTENCE`.

### output
A dictionary of corrected tokens with score

### Usage
Set the `SENTENCE` as an argument.

```bash
$ python3 bertjsc.py --input 日本語校正してい
...
 INFO arg_utils.py (13) : Start!
 INFO arg_utils.py (163) : env_id: 1
 INFO arg_utils.py (166) : VulkanDNN-Radeon RX 560 Series
 INFO bertjsc.py (89) : input_text: 日本語校正してい
 INFO bertjsc.py (92) : inference has started...
 INFO bertjsc.py (111) : corrected_tokens:
{1: {'score': 21.581863, 'token': '日本語'},
 2: {'score': 16.697094, 'token': '校'},
 3: {'score': 15.1636915, 'token': '正'},
 4: {'score': 23.816902, 'token': 'し'},
 5: {'score': 24.06585, 'token': 'て'},
 6: {'score': 16.251663, 'token': 'いる'}}
 INFO bertjsc.py (113) : Script finished successfully.
```

### Framework
PyTorch 1.6.0

### Model Format
ONNX opset = 11

### Netron

- [distilbert-base-uncased-finetuned-sst-2-english.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_sentiment_analysis/distilbert-base-uncased-finetuned-sst-2-english.onnx.prototxt)
