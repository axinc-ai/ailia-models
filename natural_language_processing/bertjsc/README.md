# Correct Japanese Sentence Using Bert

### input
A `SENTENCE`.

### output
A dictionary of corrected tokens with score

### Usage
Set the `SENTENCE` as an argument.

```bash
$ python3 bertjsc.py --input 日本語を校正しま.
...
 INFO arg_utils.py (13) : Start!
 INFO arg_utils.py (163) : env_id: 1
 INFO arg_utils.py (166) : VulkanDNN-Radeon RX 560 Series
 INFO model_utils.py (84) : ONNX file and Prototxt file are prepared!
 INFO bertjsc.py (91) : input_text: 日本語を校正しま.
 INFO bertjsc.py (94) : inference has started...
 INFO bertjsc.py (113) : corrected_tokens:
{1: {'correct': '日本語', 'score': 0.99834263, 'token': '日本語'},
 2: {'correct': 'を', 'score': 0.9902327, 'token': 'を'},
 3: {'correct': '校', 'score': 0.9638576, 'token': '校'},
 4: {'correct': '正', 'score': 0.87244976, 'token': '正'},
 5: {'correct': 'する', 'score': 0.6893229, 'token': 'しま'},
 6: {'correct': '.', 'score': 0.9992901, 'token': '.'}}
 INFO bertjsc.py (115) : Script finished successfully.
```

### Framework
PyTorch

### Model Format
ONNX opset = 11

### Netron

- [bertjsc](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bertjsc/bertjsc.onnx.prototxt)
