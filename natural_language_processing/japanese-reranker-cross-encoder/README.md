# japanese-reranker-cross-encoder-large-v1

## Input

Query
```
感動的な映画について
```

Sentences
```
深いテーマを持ちながらも、観る人の心を揺さぶる名作。登場人物の心情描写が秀逸で、ラストは涙なしでは見られない。
重要なメッセージ性は評価できるが、暗い話が続くので気分が落ち込んでしまった 。もう少し明るい要素があればよかった。
どうにもリアリティに欠ける展開が気になった。もっと深みのある人間ドラマが見たかった。
アクションシーンが楽しすぎる。見ていて飽きない。ストーリーはシンプルだが、それが逆に良い。
```

## Output

The scores in order of higher
```
(1) 深いテーマを持ちながらも、観る人の心を揺さぶる名作。登場人物の心情描写が秀逸で、ラストは涙なしでは見られない。 (0.778)
(4) アクションシーンが楽しすぎる。見ていて飽きない。ストーリーはシンプルだが、それが逆に良い。 (0.210)
(3) どうにもリアリティに欠ける展開が気になった。もっと深みのある人間ドラマが見たかった。 (0.023)
(2) 重要なメッセージ性は評価できるが、暗い話が続くので気分が落ち込んでしまった。もう少し明るい要素があればよかった。 (0.020)
```

## Requirements

This model requires additional module.

```
pip3 install transformers
pip3 install fugashi
pip3 install unidic-lite
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample sentences,
```bash
$ python3 japanese-reranker-cross-encoder.py
```

If you want to specify the text file of sentences, put the file path after the `--input` option.
```bash
$ python3 japanese-reranker-cross-encoder.py --input TEXT_PATH
```

If you want to specify an input query, write the query after the `--query` option.
```bash
$ python3 japanese-reranker-cross-encoder.py --query QUERY
```

## Reference

- [Hugging Face - japanese-reranker-cross-encoder-large-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-large-v1)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[japanese-reranker-cross-encoder-large-v1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/japanese-reranker-cross-encoder/japanese-reranker-cross-encoder-large-v1.onnx.prototxt)
