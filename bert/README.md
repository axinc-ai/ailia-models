# Predicting Missing Word Using Bert Masked LM

### input
A sentence with a masked word, which is defined as `SENTENCE` in `bert.py`.  
Masked Word should be represented by one `_`.

### output
Top `k` predicted words suitable for filling the Masked Word.  
`k` is defined as `NUM_PREDICT` in `bert.py`

### Usage
ex. SENTENCE = 'I want to _ the car because it is cheap.'

```bash
$ python3 bert.py
...
predicted top 3 words: ['buy', 'drive', 'rent']
```

### Reference
[pytorch-pretrained-bert](https://pypi.org/project/pytorch-pretrained-bert/)

### Framework
PyTorch 1.3.0

### Model Format
ONNX opset = 10

### Netron

[bert-base-uncased.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/bert_en/bert-base-uncased.onnx.prototxt)
