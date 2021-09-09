# ABD-Net: Attentive but Diverse Person Re-Identification

## Input (query)

![Input](query/0342_c5s1_079123_00.jpg)

(Image from http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip)

Shape : (batch, 3, height, width)

## Output (Top 10 images from gallery images that are similar to query)

![Output](output.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 abd_net.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 abd_net.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

If you want to specify the directory of gallery image, put the directory path after the `--gallery_dir` option.
```bash
$ python3 abd_net.py --gallery_dir gallery
```
Now, files in this gallery directory are very restricted.   
Many more files can be found in the bounding_box_test directory of [Market-1501-v15.09.15.zip](http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip) or [DukeMTMC-VideoReID.zip](https://drive.google.com/file/d/1qIadJTpY3Wpvsubui2c4jIQTUhAWA1-y/view).

Once the program run, a intermediate result file containing the features of the gallery image will be created.  
By adding the intermediate result file name after the `--data` option, you can use the intermediate result of the previous inference.
```bash
$ python3 abd_net.py --data result_resnet50.npy
```

By adding the model name after the `--model` option, you can specify the model.  
The model name is selected from 'market1501', 'duke', 'msmt17'.
```bash
$ python3 abd_net.py --model market1501
```

## Reference


- [Attentive but Diverse Person Re-Identification](https://github.com/VITA-Group/ABD-Net)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[abd_net_market1501.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/abd_net/abd_net_market1501.onnx.prototxt)  
[abd_net_duke.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/abd_net/abd_net_duke.prototxt)  
[abd_net_msmt17.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/abd_net/abd_net_msmt17.prototxt) 
