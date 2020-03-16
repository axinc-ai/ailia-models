# Predicting Missing Word Using Bert Masked LM

### input
A sentence with a masked word, which is defined as `SENTENCE` in `bert.py`.  
Masked Word should be represented by one `_`.

### output
Top `k` predicted words suitable for filling the Masked Word.  
`k` is defined as `NUM_PREDICT` in `bert.py`

### Reference
[pytorch-pretrained-bert](https://pypi.org/project/pytorch-pretrained-bert/)

### Framework
PyTorch 1.3.0

### Model Format
ONNX opset = 10
