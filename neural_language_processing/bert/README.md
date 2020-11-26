# Predicting Missing Word Using Bert Masked LM

### input
A sentence with a masked word, which is defined as `SENTENCE` in `bert.py`.  
Masked Word should be represented by one `_`.

### output
Top `k` predicted words suitable for filling the Masked Word.  
`k` is defined as `NUM_PREDICT` in `bert.py`

### Usage
`SENTENCE` is defined in the `bert.py`.
ex. SENTENCE = 'I want to _ the car because it is cheap.'

- English Bert
```bash
$ python3 bert.py
...
predicted top 3 words: ['buy', 'drive', 'rent']
```

- Japanese Bert (test implementation)
  - [WARNING] For now, Japanese model does not work correctly.
  - requirements
	- juman++, boost (ref: https://blog.imind.jp/entry/2019/01/12/192936)

```bash
$ python3 bert.py -l jp
predicted top 3 words: ['結婚', '[UNK]', '旅']
```


### Reference
[pytorch-pretrained-bert](https://pypi.org/project/pytorch-pretrained-bert/)  
[BERT日本語Pretrainedモデル](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB)

### Framework
PyTorch 1.3.0

### Model Format
ONNX opset = 10

### Netron

[bert-base-uncased.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_en/bert-base-uncased.onnx.prototxt)
