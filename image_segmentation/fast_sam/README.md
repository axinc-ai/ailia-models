# FastSAM

### Input

![input_image](cat.jpg)  

(Image from https://github.com/CASIA-IVA-Lab/FastSAM/tree/main/images/cat.png)

### Output
#### segmentation
![output_image](output.png)


#### Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 fast_sam.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 fast_sam.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--model_type`, `-m` option, you can specify model type which is selected from "FastSAM-s","FastSAM-x".(default is FastSAM-x)
```bash
$ python3 fast_sam.py -m FastSAM-s
```

#### Box
![output_image](output_box.png)

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 fast_sam.py --box_prompt "[[252,108,726,1808]]"
```

#### Text Prompt
![output_image](output_text.png)

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 fast_sam.py --text_prompt "cat" 
```

### Reference

- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)

### Framework

PyTorch

### Model Format

ONNX opset = 17

### Netron

[FastSAM-s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/fast_sam/FastSAM-s.onnx.prototxt)
[FastSAM-x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/fast_sam/FastSAM-x.onnx.prototxt)

[ViT-B32-encode_image.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/clip/ViT-B32-encode_image.onnx.prototxt)
[ViT-B32-encode_text.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/clip/ViT-B32-encode_text.onnx.prototxt)
