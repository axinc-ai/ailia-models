# Text Generation Japanese LLama Elyza

### input
A `SENTENCE`in japanese, example: クマが海辺に行ってアザラシと友達になり、最終的には家に帰るというプロットの短編小説を書いてください。

### output
A text, consisting of a number of tokens equivalent to the "--outlength" parameter, is stored as a TXT file at the specified "SAVE_PATH" location.

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample file use following command. 
```
pip3 install -r requirements.txt

python3 elyza.py 
```

You can use an orbitary sentence with:

```
python3 elyza.py --input クマが海辺に行ってアザラシと友達になり、最終的には家に帰るというプロットの短編小説を書いてください。
```

If you want to specify the maximum number of generated tockens (e.g. 300 tockens), please use following command:

```
python3 elyza.py  --outlength 300
```

if you want to run a benchmark of the model inference, you can use :

```
python3 elyza.py  --benchmark
```
### Framework
PyTorch, HuggingFace Transformers

### Model Format
ONNX opset = 13

### Netron

[Elyza LLama model](LICENSE_LLama)

- [decoder_model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/elyza-japanese-llama-2-7b/decoder_model.onnx.prototxt)