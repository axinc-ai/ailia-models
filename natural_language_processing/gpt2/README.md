# Text Generation Using GPT2

### input
A `SENTENCE`.

### output
`SENTENCE`

### Usage
Set the `SENTENCE` and the `CHARCOUNT` (number of characters to be output) in the argument.

```bash
$ python3 gpt2.py -i "My name is Clara and I am" -o 30
...
Input  :  My name is Clara and I am
Output :  My name is Clara and I am 17 weeks old baby. Today i will tell ya what the future looks very very dark with me baby! In case u got questions please give email.
```

### Reference
[GPT-2](https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/README.md)  

### Framework
PyTorch 1.7.0

### Model Format
ONNX opset = 11

### Netron

- [gpt2/gpt2-medium.onnx.opt.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt2/gpt2-medium.onnx.opt.prototxt)
