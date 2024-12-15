# Phi-3 Mini-128K-Instruct

### input
A `SENTENCE`.

### output
`SENTENCE`

### Usage
Set the `SENTENCE` in the argument.

```bash
$ python3 phi3.py -i "<|user|>Tell me a joke<|end|><|assistant|>" -o 30
...
Input  :  <|user|>Tell me a joke<|end|><|assistant|>
Output :  
```

### Reference
- [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
- [Phi-3-mini-128k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx)  
- [ONNX Runtime supports Phi-3 mini models across platforms and devices](https://onnxruntime.ai/blogs/accelerating-phi-3)  

### Framework
ONNX Runtime

### Netron

- [phi3-mini-128k-instruct-cuda-fp16.onnx](https://netron.app/?url=https://storage.googleapis.com/ailia-models/phi3/phi3-mini-128k-instruct-cuda-fp16.onnx)
