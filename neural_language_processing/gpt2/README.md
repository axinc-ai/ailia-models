# Text Generation Using GPT2

### input
A `SENTENCE`.

### output
`SENTENCE`

### Usage
Set the `SENTENCE` and the `CHARCOUNT` (number of characters to be output) in the argument.

```bash
$ python3 rinna_gpt2.py -i "My name is Clara and I am" -o 30
...
Input  :  My name is Clara and I am
Output :  My name is Clara and I am 17 weeks from conception; and it appears as being of an inferior order: the "lower and better way". This would certainly come close, given Trump
```

### Reference
[GPT-2](https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/README.md)  

### Framework
PyTorch 1.7.0

### Model Format
ONNX opset = 11

### Netron

- [gpt2/gpt2-medium.onnx.opt.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt2/gpt2-medium.onnx.opt.prototxt)
