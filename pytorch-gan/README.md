# Pytorch-ZOO GNet

## Input: no input (random input generated inside of the python script)

## Output

![Output](output.png)

## Note

The generative networks presented and used here have been trained to generate anime faces
and celebrity faces by the company ax Inc. (in 2020).

The training was performed using the framework Pytorch-ZOO GAN.
Software license agreement of Pytorch-Zoo GAN: see LICENSE file

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

To generate `output.png`,
``` bash
$ python3 ailia-pytorch-gnet.py
```

You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 ailia-pytorch-gnet.py --savepath SAVE_IMAGE_PATH
```

You can also generate using celeb model.

``` bash
$ python3 ailia-pytorch-gnet.py -m celeb
```


## Reference

[Code repo for the Pytorch GAN Zoo project (used to train this model)](https://github.com/facebookresearch/pytorch_GAN_zoo)

## Framework

Pytorch

## Model Format

ONNX opset=10

## Netron

[pytorch-gnet-animeface.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/pytorch-gan/pytorch-gnet-animeface.onnx.prototxt)

[pytorch-gnet-celeba.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/pytorch-gan/pytorch-gnet-celeba.onnx.prototxt)

