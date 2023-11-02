# PreSumm

### Input
A text file with a new line for each sentence.

### Output
```
Start summarize...
Top 1: It computes a score for each sentence separated by a newline.
Top 2: In this program, the default number of sentences to be extracted is set to 3.
Top 3: The sentences are then extracted in order of highest score.
```

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample text file,
```bash
$ python3 presumm.py
```
If you want to specify the input text file, you can use file option.
```bash
$ python3 presumm.py --file other.txt
```

### Reference
[PreSumm](https://github.com/nlpyang/PreSumm)  

### Framework
PyTorch

### Model Format
ONNX opset = 11

### Netron
- [cnndm-bertext.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/presumm/cnndm-bertext.onnx.prototxt)
