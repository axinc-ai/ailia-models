# Predicting Missing Word Using Bert Masked LM

### input
A sentence with a masked word, which is defined as `SENTENCE` in `bert_maskedlm.py`.  
Masked Word should be represented by one `_`.

### output
Top `k` predicted words suitable for filling the Masked Word.  
`k` is defined as `NUM_PREDICT` in `bert_maskedlm.py`

### Usage
Set the `SENTENCE` as an argument.

- English Bert
```bash
$ python3 bert_maskedlm.py -i "I have a [MASK]" -a bert-base-cased 
...
Input text : I have a [MASK]
Tokenized text :  ['I', 'have', 'a', '[MASK]']
Indexed tokens :  [146, 1138, 170, 103]
Predicting...
Predictions : 
0 friend
1 girl
2 man
3 love
4 woman
```

- Japanese Bert
```bash
$ python3 bert_maskedlm.py -i "私は[MASK]で動く。" -a bert-base-japanese-whole-word-masking
...
Input text : 私は[MASK]で動く。
Tokenized text :  ['私', 'は', '[MASK]', 'で', '動く', '。']
Indexed tokens :  [1325, 9, 4, 12, 11152, 8]
Predicting...
Predictions : 
0 単独
1 高速
2 自動
3 屋内
4 、
```

### Proofreeding

You can use MaskedLM to proofread your text. After masking the word and making a prediction, the part where the probability of occurrence of the original word is low is displayed in red.

- English Bert
```bash python3 bert_maskedlm_proofreeding.py -i test_text_en.txt -a bert-base-cased
...
 This program proofreads sentences .
 This program analyzes sentences to d"sa" detect typographical errors . The location of the typographical error is displayed in red .
Script finished successfully.
```

- Japanese Bert
```bash
$ python3 bert_maskedlm_proofreeding.py -i test_text_jp.txt -a bert-base-japanese-whole-word-masking
...
文章の校正のテスト
本プログラムでは文章を解析して誤植を検出しまあ"あす"。誤植の位置は赤で表示されます。
```


### Reference
[transformers](https://github.com/huggingface/transformers)  

### Framework
PyTorch 1.6.0

### Model Format
ONNX opset = 11

### Netron

- [bert-base-cased.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_maskedlm/bert-base-cased.onnx.prototxt)
- [bert-base-uncased.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_maskedlm/bert-base-uncased.onnx.prototxt)
- [bert-base-japanese-whole-word-masking](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_maskedlm/bert-base-japanese-whole-word-masking.onnx.prototxt)
