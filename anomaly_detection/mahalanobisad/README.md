# Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection

## Input

<img src="bottle_test_broken_large_000.png" width="256" height="256">

(Image from MVTec AD datasets https://www.mvtec.com/company/research/datasets/mvtec-ad/)

Input shape : (n, 3, 224, 224)

<br/>

## Output

Left to right: input, ground truth, predicted mask, segmentation result

<img src=https://github.com/user-attachments/assets/faec20a2-2702-4747-b7da-2b191c38bf92 height="256">
<img src=https://github.com/user-attachments/assets/f8f7b3a0-d0ad-46fa-9d6e-9d84de0a3fd0 height="256">
<img src=https://github.com/user-attachments/assets/f7b7dfa3-d11a-4368-9084-e5bc7e463ee3 height="256">
<img src=https://github.com/user-attachments/assets/78c7df66-692b-4ef0-b5ee-ec0ac11e0b2a height="256">

<br/>
<br/>

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

In order to get the feature vector of the normal product, it is necessary to prepare the file of the normal product.  
By default, normal files are got from the `train` directory.  

To calculate the threshold you also need to prepare some test images and corresponding ground truth images.
By default, test images are taken from the `images` directory and ground truth images are taken from the `gt_masks` directory.

For the sample image, first download [MVTec AD datasets](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and place `bottle/train/good/*.png` files to the `train` directory.  

For the sample image.
```bash
$ python3 mahalanobisad.py
```

<br/>

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 mahalanobisad.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

<br/>

The feature vectors created from files in the train directory are saved to the pickle file.  
From the second time, by specifying the pickle file by `--feat` option,
it can omit the calculation of the feature vector of the normal product.  
The name of the pickle file created is the same as the name of a normal product file directory.
```bash
$ python3 mahalanobisad.py --feat train.pkl
```

<br/>

You can specify the directory of normal product files with the `--train_dir` option.
```bash
$ python3 mahalanobisad.py --train_dir train
```

<br/>

The threshold is calculated using the test images and the ground truth images, but you can also give a threshold.
```bash
$ python3 mahalanobisad.py --threshold 300
```

<br/>

By adding the `--aug` option, you can process with augmentation.  
(default is processing without augmentation)
```bash
$ python3 mahalanobisad.py --aug
```

<br/>

By adding the `--aug_num` option, you can specify amplification factor of data by augmentation.
(default is 5)
```bash
$ python3 mahalanobisad.py --aug --aug_num 5
```

<br/>

## Reference

[MahalanobisAD-pytorch](https://github.com/byungjae89/MahalanobisAD-pytorch/tree/master)

<br/>

## Framework

Pytorch

<br/>

## Model Format

ONNX opset=11

<br/>

## Netron

[efficientnet-b0.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mahalanobisad/efficientnet-b0.onnx.prototxt)  
[efficientnet-b1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mahalanobisad/efficientnet-b1.onnx.prototxt)  
[efficientnet-b2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mahalanobisad/efficientnet-b2.onnx.prototxt)  
[efficientnet-b3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mahalanobisad/efficientnet-b3.onnx.prototxt)  
[efficientnet-b4.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mahalanobisad/efficientnet-b4.onnx.prototxt)  
[efficientnet-b5.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mahalanobisad/efficientnet-b5.onnx.prototxt)  
[efficientnet-b6.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mahalanobisad/efficientnet-b6.onnx.prototxt)  
[efficientnet-b7.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mahalanobisad/efficientnet-b7.onnx.prototxt)
