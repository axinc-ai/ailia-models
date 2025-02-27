# Zero Shot Classification in Japanese

### input
A `SENTENCE`, `CANDIDATE_LABELS` and `HYPOTHESIS_TEMPLATE`.

### output
Probability scores for each label.

### Usage
Set the `SENTENCE`, `CANDIDATE_LABELS` and `HYPOTHESIS_TEMPLATE` as an argument.

```bash
$ python3 zero_shot_classification_japanese.py -s "今日、新しいiPhoneが発売されました" -c "スマートフォン, エンタメ, スポーツ" -t "This example is {}."
...
+ idx=0
  category=0[スマートフォン ]
  prob=0.8609882593154907
+ idx=1
  category=1[エンタメ ]
  prob=0.1195221021771431
+ idx=2
  category=2[スポーツ ]
  prob=0.019489625468850136
```

You can select a model from `minilm_l6 | minilm_l12` by adding --arch (default: minilm_l12).

### Reference
- [MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli)
- [MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli)

### Framework
PyTorch

### Model Format
ONNX opset = 14

### Netron
[minilm_l12.onnx](https://netron.app/?url=https://storage.googleapis.com/ailia-models/zero_shot_classification_japanese/minilm_l12.onnx)  
[minilm_l6.onnx](https://netron.app/?url=https://storage.googleapis.com/ailia-models/zero_shot_classification_japanese/minilm_l12.onnx)  
