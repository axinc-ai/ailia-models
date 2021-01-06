# Style2Paints

## Input

![Input](Apr19H22M03S00R696.jpg)

(Image from https://github.com/lllyasviel/style2paints/blob/master/V3/server/game/samples/Apr19H22M03S00R696/sketch.original.jpg)

## Output

![Output](output.png)

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,

``` bash
$ python3 style2paints.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 style2paints.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

A json file with the same name as IMAGE_PATH will be read as an option file.
The option file has the following contents.

```buildoutcfg
{"alpha":0,"points":[[0.44533212358531116,0.7223775612739742,255,247,231,2],[0.4298600830905879,0.6796168465201625,255,247,231,2],[0.5347261353326012,0.6825658613307702,255,247,231,2],[0.23216178788023506,0.5852483725807156,255,247,231,2],[0.17371185712239157,0.4746603171829263,255,247,231,2],[0.14276777613294506,0.38618987286469475,255,247,231,2],[0.1376104293013706,0.3272095766525405,255,247,231,2],[0.23903825032233428,0.3360566210843637,255,247,231,2],[0.5656702163220477,0.41125649875486037,255,247,231,2],[0.5398834821641756,0.3404801433002752,255,247,231,2],[0.589737834869395,0.23726462492900516,255,247,231,2],[0.6206819158588416,0.1812333435274586,255,247,231,2],[0.6395921875746144,0.24758617676613215,255,247,231,2],[0.3146793371854258,0.5129975097208265,255,255,254,2],[0.2545102908170575,0.4304250950238106,255,255,254,2],[0.4539277016379353,0.4805583468041418,255,255,254,2],[0.33187049329067386,0.35522521735331375,255,255,254,2],[0.3954777708800918,0.2726528026562977,255,255,254,2],[0.47455708896423293,0.25643322119795536,255,255,254,2],[0.47971443579580736,0.14584516580016602,255,255,254,2],[0.4865908982379066,0.03968063261828827,255,255,254,2],[0.41094981137481507,0.06622176591375764,255,255,254,2],[0.6034907597535935,0.026410065970553544,255,255,254,2],[0.661940690511437,0.2991939359517672,255,255,254,2],[0.6791318466166851,0.3714447988116562,255,255,254,2],[0.46080416408003444,0.39356240989121416,255,20,147,2],[0.42470273625901356,0.3611232469745292,255,20,147,2],[0.4659615109116089,0.43189960242911435,255,20,147,2],[0.5020629387326299,0.39946043951242943,255,20,147,2],[0.5020629387326299,0.3390056358949713,255,20,147,2],[0.5519172914378493,0.7887303945126479,75,0,130,0],[0.4487703548063608,0.09423740661453096,255,255,255,0],[0.6241201470798912,0.8993184499104372,220,176,211,2],[0.7031994651640323,0.830016601861156,220,176,211,2],[0.7633685115324006,0.695836427978505,220,176,211,2],[0.563951100711523,0.49087989864126874,255,222,173,0],[0.44361300797478637,0.8491851981301061,255,255,255,2],[0.49518647629053064,0.8654047795884485,255,255,255,2],[0.5381643665536509,0.8742518240202717,255,255,255,2],[0.5931760660904447,0.8639302721831448,255,255,255,2],[0.651625996848288,0.8432871685088906,255,255,255,2]],"method":"colorization","lineColor":[0,0,0],"line":false,"hasReference":false}
```

(File from https://github.com/lllyasviel/style2paints/blob/master/V3/server/game/samples/Apr19H22M03S00R696/options.sample.json)

The most popular parameter is "points" that indicate the "draft color point" and the "accurate color point".
This is a list of array has the values of "x y r g b t".  
The "x" and "y" are shown as a percentage of the width and height of image, with the lower left being the origin.  
If the "t" is 0 indicates a "draft color point", and the "t" is 2 indicates an "accurate color point".  
The "method" paramter is selected from "colorization", "rendering", "recolorization".

## Reference

- [Style2Paints](https://github.com/lllyasviel/style2paints)

## Framework

Tensorflow

## Model Format

ONNX opset=11

## Netron

[head.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/style2paints/head.onnx.prototxt)
[neck.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/style2paints/neck.onnx.prototxt)
[baby.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/style2paints/baby.onnx.prototxt)
[tail.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/style2paints/tail.onnx.prototxt)
[gird.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/style2paints/gird.onnx.prototxt)
