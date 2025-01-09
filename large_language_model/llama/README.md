# Text Generation Using LLaMA and RWKV

## LLaMA (Weight size 26GB for FP32, 13GB for FP16)

### input
A `SENTENCE`.

### output
`SENTENCE`

### Usage
Set the `SENTENCE` and the `CHARCOUNT` (number of characters to be output) in the argument.

```bash
$ python3 llama.py -i "bonjour" 
...
Output :  bonjour
```

```bash
$ python3 llama.py -i "bonjour" --fp16
...
Output :  bonjour
```

## RWKV (Weight size 920MB)

### input
A `SENTENCE`.

### output
`SENTENCE`

### Usage
Set the `SENTENCE` and the `CHARCOUNT` (number of characters to be output) in the argument.

```bash
$ python llama.py  -m rwkv  -i "\nIn a shocking finding," --length 100

...
output :  researchers have found that the majority of people who have been exposed to the coronavirus are not infected with the virus.

The study, published in the journal Nature, found that the majority of people who have been exposed to the coronavirus are not infected with the virus.

The study, published in the journal Nature, found that the majority of people who have been exposed to the coronavirus are not infected with the virus.

The study, published in the journal Nature, found that the majority of
```

### Reference
[LLaMa/RWKV onnx](https://github.com/tpoisonooo/llama.onnx)  

### Framework
pytorch 2.0.0

### Model Format
ONNX opset = 14

### Netron

- [embed.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/embed.onnx.prototxt)
- [head.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/head.onnx.prototxt)
- [norm.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/norm.onnx.prototxt)
- [decoder-merge-0.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-0.onnx.prototxt)
- [decoder-merge-1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-1.onnx.prototxt)
- [decoder-merge-2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-2.onnx.prototxt)
- [decoder-merge-3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-3.onnx.prototxt)
- [decoder-merge-4.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-4.onnx.prototxt)
- [decoder-merge-5.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-5.onnx.prototxt)
- [decoder-merge-6.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-6.onnx.prototxt)
- [decoder-merge-7.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-7.onnx.prototxt)
- [decoder-merge-8.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-8.onnx.prototxt)
- [decoder-merge-9.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-9.onnx.prototxt)
- [decoder-merge-10.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-10.onnx.prototxt)
- [decoder-merge-11.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-11.onnx.prototxt)
- [decoder-merge-12.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-12.onnx.prototxt)
- [decoder-merge-13.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-13.onnx.prototxt)
- [decoder-merge-14.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-14.onnx.prototxt)
- [decoder-merge-15.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-15.onnx.prototxt)
- [decoder-merge-16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-16.onnx.prototxt)
- [decoder-merge-17.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-17.onnx.prototxt)
- [decoder-merge-18.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-18.onnx.prototxt)
- [decoder-merge-19.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-19.onnx.prototxt)
- [decoder-merge-20.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-20.onnx.prototxt)
- [decoder-merge-21.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-21.onnx.prototxt)
- [decoder-merge-22.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-22.onnx.prototxt)
- [decoder-merge-23.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-23.onnx.prototxt)
- [decoder-merge-24.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-24.onnx.prototxt)
- [decoder-merge-25.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-25.onnx.prototxt)
- [decoder-merge-26.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-26.onnx.prototxt)
- [decoder-merge-27.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-27.onnx.prototxt)
- [decoder-merge-28.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-28.onnx.prototxt)
- [decoder-merge-29.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-29.onnx.prototxt)
- [decoder-merge-30.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-30.onnx.prototxt)
- [decoder-merge-31.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-31.onnx.prototxt)
- [embed_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/embed_fp16.onnx.prototxt)
- [head_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/head_fp16.onnx.prototxt)
- [norm_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/norm_fp16.onnx.prototxt)
- [llama/decoder-merge-0_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-0_fp16.onnx.prototxt)
- [llama/decoder-merge-1_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-1_fp16.onnx.prototxt)
- [llama/decoder-merge-2_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-2_fp16.onnx.prototxt)
- [llama/decoder-merge-3_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-3_fp16.onnx.prototxt)
- [llama/decoder-merge-4_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-4_fp16.onnx.prototxt)
- [llama/decoder-merge-5_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-5_fp16.onnx.prototxt)
- [llama/decoder-merge-6_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-6_fp16.onnx.prototxt)
- [llama/decoder-merge-7_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-7_fp16.onnx.prototxt)
- [llama/decoder-merge-8_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-8_fp16.onnx.prototxt)
- [llama/decoder-merge-9_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-9_fp16.onnx.prototxt)
- [llama/decoder-merge-10_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-10_fp16.onnx.prototxt)
- [llama/decoder-merge-11_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-11_fp16.onnx.prototxt)
- [llama/decoder-merge-12_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-12_fp16.onnx.prototxt)
- [llama/decoder-merge-13_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-13_fp16.onnx.prototxt)
- [llama/decoder-merge-14_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-14_fp16.onnx.prototxt)
- [llama/decoder-merge-15_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-15_fp16.onnx.prototxt)
- [llama/decoder-merge-16_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-16_fp16.onnx.prototxt)
- [llama/decoder-merge-17_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-17_fp16.onnx.prototxt)
- [llama/decoder-merge-18_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-18_fp16.onnx.prototxt)
- [llama/decoder-merge-19_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-19_fp16.onnx.prototxt)
- [llama/decoder-merge-20_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-20_fp16.onnx.prototxt)
- [llama/decoder-merge-21_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-21_fp16.onnx.prototxt)
- [llama/decoder-merge-22_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-22_fp16.onnx.prototxt)
- [llama/decoder-merge-23_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-23_fp16.onnx.prototxt)
- [llama/decoder-merge-24_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-24_fp16.onnx.prototxt)
- [llama/decoder-merge-25_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-25_fp16.onnx.prototxt)
- [llama/decoder-merge-26_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-26_fp16.onnx.prototxt)
- [llama/decoder-merge-27_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-27_fp16.onnx.prototxt)
- [llama/decoder-merge-28_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-28_fp16.onnx.prototxt)
- [llama/decoder-merge-29_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-29_fp16.onnx.prototxt)
- [llama/decoder-merge-30_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-30_fp16.onnx.prototxt)
- [llama/decoder-merge-31_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/decoder-merge-31_fp16.onnx.prototxt)
- [embed_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/embed_rwkv.onnx.prototxt)
- [head_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/head_rwkv.onnx.prototxt)
- [mixing-0_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_0_rwkv.onnx.prototxt)
- [mixing-1_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_1_rwkv.onnx.prototxt)
- [mixing-2_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_2_rwkv.onnx.prototxt)
- [mixing-3_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_3_rwkv.onnx.prototxt)
- [mixing-4_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_4_rwkv.onnx.prototxt)
- [mixing-5_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_5_rwkv.onnx.prototxt)
- [mixing-6_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_6_rwkv.onnx.prototxt)
- [mixing-7_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_7_rwkv.onnx.prototxt)
- [mixing-8_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_8_rwkv.onnx.prototxt)
- [mixing-9_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_9_rwkv.onnx.prototxt)
- [mixing-10_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_10_rwkv.onnx.prototxt)
- [mixing-11_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_11_rwkv.onnx.prototxt)
- [mixing-12_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_12_rwkv.onnx.prototxt)
- [mixing-13_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_13_rwkv.onnx.prototxt)
- [mixing-14_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_14_rwkv.onnx.prototxt)
- [mixing-15_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_15_rwkv.onnx.prototxt)
- [mixing-16_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_16_rwkv.onnx.prototxt)
- [mixing-17_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_17_rwkv.onnx.prototxt)
- [mixing-18_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_18_rwkv.onnx.prototxt)
- [mixing-19_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_19_rwkv.onnx.prototxt)
- [mixing-20_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_20_rwkv.onnx.prototxt)
- [mixing-21_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_21_rwkv.onnx.prototxt)
- [mixing-22_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_22_rwkv.onnx.prototxt)
- [mixing-23_rwkv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llama/mixing_23_rwkv.onnx.prototxt)
