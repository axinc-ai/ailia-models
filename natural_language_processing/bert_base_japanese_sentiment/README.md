# View Adaptive Neural Networks (VA) for Skeleton-based Human Action Recognition

## Input

A SENTENCE.

- Sample
```
私は幸せである。
```

## Output

Recognized emotions
```
ポジティブ : 0.9903476238250732
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample sentence,
```bash
$ python3 bert_base_japanese_sentiment.py
```

If you want to specify the `SENTENCE`, put after the `--input` option.
```bash
$ python3 bert_base_japanese_sentiment.py --input SENTENCE
```

## Reference

[Hugging Face - daigo/bert-base-japanese-sentiment](https://huggingface.co/daigo/bert-base-japanese-sentiment)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_base_japanese_sentiment/model.onnx.prototxt)
