# pytorch-superpoint

## Input

- img_A

![Input](img_A.png)

- img_B

![Input](img_B.png)

(Image from http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)

## Output

- Keypoints result

![Output](output_keypoints.png)

- Matches result (inliers)

![Output](output_match_inliers.png)

- Matches result (outliers)

![Output](output_match_outliers.png)

- Warping result (gray)

![Output](output_warping_gray.png)

- Warping result (filterd)

![Output](output_warping_filtered.png)

- Warping result (correspondence)

![Output](output_warping_correspondence.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 pytorch_superpoint.py
```

If you want to specify the input image, put the image path (as img_B) after the `--input` option, 
and the second image path (as img_A) after the `--input2` option.  
```bash
$ python3 pytorch_superpoint.py
 --input IMAGE_B --input2 IMAGE_A
```

## Reference

- [pytorch-superpoint](https://github.com/eric-yyjau/pytorch-superpoint)

- [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/pdf/1712.07629.pdf)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[SuperPointNet_gauss2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/pytorch-superpoint/SuperPointNet_gauss2.onnx.prototxt)
