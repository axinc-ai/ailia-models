# Leveraging BERT for Extractive Text Summarization on Lectures

### Input
A text file with a new line for each sentence.

### Output
Top `NUM_PREDICTS` extracted summary statements.  
`NUM_PREDICTS` is defined in `bert_sum_ext.py`

### Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample text file,
```bash
$ python3 bert_sum_ext.py
```
If you want to specify the input text file, put the text file path after the -f option.
```bash
$ python3 bert.py -f other.txt
```


### Reference
[BERT Extractive Summarizer](https://github.com/dmmiller612/bert-extractive-summarizer)  
[日本語BERT](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB)

### Framework
PyTorch

### Model Format
ONNX opset = 11

### Netron

[bert-base.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_sum_ext/bert-base.onnx.prototxt)
