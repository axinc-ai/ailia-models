# GLuCoSE (General Luke-based COntrastive Sentence Embedding)-base-Japanese

## Input

Sentences
```
PKSHA Technologyは機械学習/深層学習技術に関わるアルゴリズムソリューションを展開している。
この深層学習モデルはPKSHA Technologyによって学習され、公開された。
広目天は、仏教における四天王の一尊であり、サンスクリット語の「種々の眼をした者」を名前の由来とする。
```

## Output

Top similar
```
#1 & #2 : 0.7544880015754691
#1 & #3 : 0.14178459123635234
#2 & #3 : 0.11260386334535319
```

## Requirements

This model requires additional module if you want to load pdf file.

```
pip3 install transformers
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample sentences,
```bash
$ python3 glucose.py
```

If you want to specify the text file of sentences, put the file path after the `--input` option.
```bash
$ python3 glucose.py --input TEXT_PATH
```

## Reference

- [Hugging Face - GLuCoSE-base-ja](https://huggingface.co/pkshatech/GLuCoSE-base-ja)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[GLuCoSE-base-ja.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glucose/GLuCoSE-base-ja.onnx.prototxt)
