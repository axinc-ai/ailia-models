# Sentiment Analysis Using Bert

### input
A `SENTENCE`.

### output
Positive or Negative

### Usage
Set the `SENTENCE` as an argument.

```bash
$ python3 bert_sentiment_analysis.py -i "Transformers and ailia SDK is an awesome combo!"
...
Input :  Transformers and ailia SDK is an awesome combo!
Label :  positive
Score :  0.99984145
```

### Reference
[transformers](https://github.com/huggingface/transformers)  

[onnx-transformers](https://github.com/patil-suraj/onnx_transformers)  

### Framework
PyTorch 1.6.0

### Model Format
ONNX opset = 11

### Netron

- [distilbert-base-uncased-finetuned-sst-2-english.onnx.prototxt(https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_sentiment_analysis/distilbert-base-uncased-finetuned-sst-2-english.onnx.prototxt)
