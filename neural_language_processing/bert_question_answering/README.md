# Question Answering Using Bert

### input
`QUESTION` and `CONTEXT`.

### output
Position of contextual text corresponding to the answer.


### Usage
Set the `QUESTION` and `CONTEXT` as an argument.

```bash
$ python3 bert_question_answering.py -q "What is ailia SDK ?" -c "ailia SDK is a highly performant single inference engine for multiple platforms and hardware"
...
Answer :  [{'score': 0.5031098127365112, 'start': 13, 'end': 91, 'answer': 'a highly performant single inference engine for multiple platforms and hardware'}]
```

### Reference
[transformers](https://github.com/huggingface/transformers)  

[onnx-transformers](https://github.com/patil-suraj/onnx_transformers)  

### Framework
PyTorch 1.6.0

### Model Format
ONNX opset = 11

### Netron

- [roberta-base-squad2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_question_answering/roberta-base-squad2.onnx.prototxt)
