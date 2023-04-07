# Leveraging BERT for Extractive Text Summarization on Lectures for Japanese

Get a feature vector for each sentence with Japanese BERT from the text. Cluster the feature vectors. Display the cluster center points as a summary.

### Requirements

```
pip3 install spacy sudachipy sudachidict_core
```

### Input
A text file with a new line for each sentence.

[sample.txt](sample.txt)

### Output

```
基盤モデルの概要
基盤モデル（Foundation Model）とは、大量のデータから学習することで、高い汎化性能を獲得したAIのことです。
特に、基盤モデルはデータセットが巨大であるため、ConvolutionよりもVision Transformerを使用する方が性能が高くなっています。
当面、エッジでの計算リソースの関係で、基盤モデルの活用は限定的になる可能性もありますが、計算リソースはハードウェアの進化と共に、増加していくため、どこかのタイミングで基盤モデルが席巻するものと考えられます。
```

Top `NUM_PREDICTS` extracted summary statements.  
`NUM_PREDICTS` is defined in `bert_sum_ext.py`

### Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample japanese text file,
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
