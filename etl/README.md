# Japanese Character Classification

### input

![input_image](font.png)

Shape: (1, 1, 28, 28) Range:[0, 1]

### output

[3189 character](etl_BINARY_squeezenet128_20.txt)

```
+ idx=0
0
3151
  category=0[ あ
 ]
  prob=0.9986122846603394
+ idx=1
4
3151
  category=4[ お
 ]
  prob=0.0010359695879742503
+ idx=2
59
3151
  category=59[ め
 ]
  prob=0.0003514652489684522
```

### Dataset

[ETL Dataset](http://etlcdb.db.aist.go.jp/?lang=ja)

### Framework

Keras

### Model Format

CaffeModel

### Netron

[etl_BINARY_squeezenet128_20.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/etl/etl_BINARY_squeezenet128_20.prototxt)
