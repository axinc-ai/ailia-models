# G2PW: A Neural Grapheme-to-Phoneme Converter for Mandarin Chinese

G2PWは、中国語（北京語）の書記素（漢字）を音素に変換するニューラルネットワークモデルです。

## Input

```
你好世界
```

## Output

**pinyin style**
```
[ni3, hao3, shi4, jie4]
```

**bopomofo style**
```
[ㄋㄧ3, ㄏㄠ3, ㄕ4, ㄐㄧㄝ4]
```

## Usage

デフォルトのテキストを使用する場合、
```bash
$ python3 g2pw.py
```

任意のテキストを指定する場合、
```bash
$ python3 g2pw.py --input '你好世界'
```

出力スタイルを指定する場合、
```bash
$ python3 g2pw.py --style pinyin
```

## Reference

- [G2PW]

## Model Format

ONNX opset=12


## Python version
Python 3.6.13 