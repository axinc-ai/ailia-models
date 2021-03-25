# PaDiM-Anomaly-Detection-Localization-master

## Input

<img src="bottle_000.png" width="256" height="256">

(Image from MVTec AD datasets https://www.mvtec.com/company/research/datasets/mvtec-ad/)

Shape : (n, 3, 244, 244)

## Output

Left to right: input, ground truth, predicted head map, predicted mask, segmentation result

![Output](output.png)

## Usage
In order to get the feature vector of the normal product, it is necessary to prepare the file of the normal product.  
By default, normal files are got from the `train` directory.  
For the sample image, first download [MVTec AD datasets](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and place `bottle/train/good/*.png` files to the `train` directory. 

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 padim.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 padim.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

The feature vectors created from files in the train directory are saved to the pickle file.  
From the second time, by specifying the pickle file by `--feat` option,
it can omit the calculation of the feature vector of the normal product.  
The name of the pickle file created is the same as the name of a normal product file directory.
```bash
$ python3 padim.py --feat train.pkl
```

You can specify the directory of normal product files with the `--train_dir` option.
```bash
$ python3 padim.py --train_dir train
```

The ground truth files are got from the `gt_masks` directory by default.  
The name of the ground truth file corresponds to the file with `__mask` after the name of the input file.  
You can specify the directory of ground truth files with the `--gt_dir` option.
```bash
$ python3 padim.py --gt_dir gt_masks
```

For bottle in MVTec AD datasets.
```bash
$ python3 padim.py -i bottle/test/broken_large --train_dir bottle/train/good --gt_dir bottle/test/ground_truth/broken_large
```

By adding the `--arch` option, you can specify model type which is selected from "resnet18", "wide_resnet50_2".  
(default is resnet18)
```bash
$ python3 padim.py --arch wide_resnet50_2
```

## Reference

[PaDiM-Anomaly-Detection-Localization-master](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[resnet18.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/padim/resnet18.onnx.prototxt)

[wide_resnet50_2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/padim/wide_resnet50_2.onnx.prototxt)
