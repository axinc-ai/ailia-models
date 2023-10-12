# Insert punctuations to Japanese sentences using BERT

### input
A `SENTENCE` without punctuation.

### output
without argument -sc: A text with punctuations inserted.
with argument -sc: A dictionary of predicted punctuations with score.

### Usage
Set the `SENTENCE` as an argument.
Set argument -sc to visualise the scores of predicted punctuations.

```bash
 $ python3 bert_insert_punctuation.py --input このモデルは音声認識モデルによって書き起こされた句読点のない文章に句読点を挿入するモデルです最大512トークン長の入力に対応しています
'''
 INFO arg_utils.py (13) : Start!
 INFO arg_utils.py (163) : env_id: 2
 INFO arg_utils.py (166) : VulkanDNN-AMD Radeon(TM) Graphics
 INFO punctbert.py (92) : input_text: このモデルは音声認識モデルによって書き起こされた句読点のない文章に句読点を挿入するモデルです最大512トークン長の入力に対応しています
 INFO punctbert.py (95) : inference has started...
 INFO punctbert.py (118) : Text with added punctuations:
'このモデルは、音声認識モデルによって書き起こされた、句読点のない文章に句読点を挿入するモデルです。最大512トークン長の入力に対応しています。'
 INFO punctbert.py (120) : Script finished successfully.
```

### Framework
PyTorch

### Model Format
ONNX opset = 11

### Netron

- [punctbert](https://netron.app/?url=)
