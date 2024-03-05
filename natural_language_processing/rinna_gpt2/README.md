# Text Generation Using rinna GPT2

### input
A `SENTENCE`.

### output
`SENTENCE`

### Usage
Set the `SENTENCE` and the `CHARCOUNT` (number of characters to be output) in the argument.

```bash
$ python3 rinna_gpt2.py -i "生命、宇宙、そして万物についての究極の疑問の答えは" -o 30
...
Input  :  生命、宇宙、そして万物についての究極の疑問の答えは
Output :  生命、宇宙、そして万物についての究極の疑問の答えは人間や宇宙そのものに存在していて、「人類に与えられています限り無限に有す」と言う「生命の法則▁人間は生まれなければ生命はないのか
```

### Reference
[japanese-pretrained-models](https://github.com/rinnakk/japanese-pretrained-models)  

### Framework
PyTorch 1.7.0

### Model Format
ONNX opset = 11

### Netron

- [rinna_gpt2/japanese-gpt2-small.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rinna_gpt2/japanese-gpt2-small.opt.onnx.prototxt)
