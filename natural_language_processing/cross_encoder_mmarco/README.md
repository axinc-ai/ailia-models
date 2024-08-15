# Cross-Encoder for multilingual MS Marco

The model can be used for Information Retrieval: Given a query, encode the query will all possible passages (e.g. retrieved with ElasticSearch). Then sort the passages in a decreasing order.

### input
`Query` and `Paragraph`.

### output
`Logits`

### Usage
Set the `Query` and `Paragraph` as an argument.

```bash
$ python3 cross_encoder_mmarco.py -q "How many people live in Berlin?" -p "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."
$ python3 cross_encoder_mmarco.py -q "How many people live in Berlin?" -p "New York City is famous for the Metropolitan Museum of Art."
$ python3 cross_encoder_mmarco.py -q "ベルリンには何人が住んでいますか？" -p "ベルリンの人口は891.82平方キロメートルの地域に登録された住民が3,520,031人います。"
$ python3 cross_encoder_mmarco.py -q "ベルリンには何人が住んでいますか？" -p "ニューヨーク市はメトロポリタン美術館で有名です。"
```

```bash
Output : [array([[10.761541]], dtype=float32)]
Output : [array([[-8.127746]], dtype=float32)]
Output : [array([[9.374646]], dtype=float32)]
Output : [array([[-6.408309]], dtype=float32)]
```

### Reference
[jeffwan/mmarco-mMiniLMv2-L12-H384-v](https://huggingface.co/jeffwan/mmarco-mMiniLMv2-L12-H384-v1)  

### Framework
- PyTorch 2.2.1
- Transformers 4.33.3

### Model Format
ONNX opset = 11

### Tokenizer
XLMRobertaTokenizer (Same with SentenceTransformer and E5)

### Netron

- [mmarco-mMiniLMv2-L12-H384-v1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/cross_encoder_mmarco/mmarco-mMiniLMv2-L12-H384-v1.onnx.prototxt)
