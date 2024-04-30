# HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.

### Input
A wav file or npy file containng mel spectograms.

### Output
The Voice file is output as .wav which path is defined as `SAVE_WAV_PATH` in `hifigan.py`.  

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample file use following command. It uses numpy file by defoult. 
```
pip3 install -r requirements.txt

python3 hifigan.py 
```

You can change type of the sample input to wav file by using following command:

```
python3 hifigan.py --inputType wav
```

If you want to specify the input file, put the wav or numpy file path after the --input option.
You can use --savepath option to change the name of the output file to save. No need to specify input type.

```
python3 hifigan.py --input test.wav --savepath SAVE_WAV_PATH
```


### Framework
PyTorch

### Model Format
ONNX opset = 11

### Netron

[HIFI GAN model](LICENSE_HIFI)

- [generator_dynamic.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hifigan/generator_dynamic.onnx.prototxt)
