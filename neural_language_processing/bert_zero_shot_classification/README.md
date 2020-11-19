# Zero Shot Classification Using Bert

### input
A `SENTENCE` and `CANDIDATE_LABELS`.

### output
Positive or Negative

### Usage
Set the `SENTENCE` and `CANDIDATE_LABELS` as an argument.

```bash
$ python3 bert_zero_shot_classification.py -s "Who are you voting for in 2020?" -c "economics, politics, public health"
...
```

### Reference
[transformers](https://github.com/huggingface/transformers)  

[onnx-transformers](https://github.com/patil-suraj/onnx_transformers)  

### Framework
PyTorch 1.6.0

### Model Format
ONNX opset = 11

### Netron

- [roberta-large-mnli.onnx](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_zero_shot_classification/roberta-large-mnli.onnx)
