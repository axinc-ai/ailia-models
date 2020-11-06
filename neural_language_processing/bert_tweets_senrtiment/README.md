# BERT tweets sentiment

### input
A twitter sentence

### output
Determine if the tweet related to the entered home appliance is positive or negative

### Usage

```bash
$ python3 bert_tweets_sentiment.py -i "iPhone 12 mini が欲しい"
Label :  positive
$ python3 bert_tweets_sentiment.py -i "iPhone 12 mini は高い"
Label :  negative
```

### Reference
[Twitter日本語評判分析データセット](http://www.db.info.gifu-u.ac.jp/data/Data_5d832973308d57446583ed9f)

### Framework
PyTorch 1.6.0

### Model Format
ONNX opset = 11

### Netron

[bert-base-uncased.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_en/bert-base-uncased.onnx.prototxt)
