# vehicle-attributes-recognition-barrier

## Input

![Input](demo.png)

(Image
from https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-attributes-recognition-barrier-0042/assets/vehicle-attributes-recognition-barrier-0042-1.png)

Shape: (1, 72, 72, 3) BGR channel order

Car pose should be front facing cars.

## Output

Estimating vehicle type and color
```bash
### Estimating vehicle type and color ###
- Type: car
- Color: black
```

### Color list

```
COLOR_LIST = (
    'white', 'gray', 'yellow', 'red', 'green', 'blue', 'black'
)
```

### Type list

```
TYPE_LIST = (
    'car', 'van', 'truck', 'bus'
)
```

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet
while downloading.

For the sample image,
``` bash
$ python3 vehicle-attributes-recognition-barrier.py 
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 vehicle-attributes-recognition-barrier.py --input IMAGE_PATH
```

## Reference

- [OpenVINO - Open Model Zoo repository - vehicle-attributes-recognition-barrier-0042](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-attributes-recognition-barrier-0042)
- [OpenVINO - vehicle-attributes-recognition-barrier-0042](https://docs.openvinotoolkit.org/latest/omz_models_model_vehicle_attributes_recognition_barrier_0042.html)

## Framework

OpenVINO

## Model Format

ONNX opset = 11

## Netron

[vehicle-attributes-recognition-barrier-0042.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vehicle-attributes-recognition-barrier/vehicle-attributes-recognition-barrier-0042.onnx.prototxt)
