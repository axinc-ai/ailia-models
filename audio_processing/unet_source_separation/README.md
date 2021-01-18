# source_separation

### input

- Noisy speech (audio file) 

```
Audio from creative commons youtube videos
https://drive.google.com/drive/folders/19Sn6pe5-BtWXYa6OiLbYGH7iCU-mzB8j
doublenoble_k7rain_part.wav
(Original video : https://www.youtube.com/watch?v=vsjB1xTwZ20&t=536s)
```

- Music (audio file)
```
Audio from SigSep Datasets
https://sigsep.github.io/datasets/dsd100.html
049 - Young Griffo - Facade.wav
(Original file : DSD100subset/Mixtures/Test/049 - Young Griffo - Facade/mixture.wav)
```

### output

Separated voice (audio file)
```
separated_voice.wav
```

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample audio file,
```bash
$ python3 source_separation.py

```

If you want to specify the input image, put the input path after the --input option.
You can use --savepath option to change the name of the output file to save.
```bash
$ python3 source_separation.py --input WAV_PATH --savepath SAVE_WAV_PATH
```

You can select a pretrained model by specifying --arch base (default) or --arch large.
`base` is model for general voice separation task, and `large` is model for singing voice separation taks.  
```bash
$ python3 source_separation.py --input WAV_PATH --savepath SAVE_WAV_PATH --arch base
```


### Reference

[source_separation](https://github.com/AppleHolic/source_separation)  

### Framework

PyTorch 1.6.0

### Model Format

ONNX opset = 11

### Netron

[second_voice_bank.best.opt.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/source_separation/second_voice_bank.best.opt.onnx.prototxt)
[RefineSpectrogramUnet.best.opt.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/source_separation/RefineSpectrogramUnet.best.opt.onnx.prototxt)
