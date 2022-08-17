# ActionCLIP

## Input

<img src="action_recognition.gif" width="480px">

(GIF from https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/action_recognition_demo/python/action_recognition.gif)

### TextCLIP

- ailia input shape: (16 \* n_text_labels, 77)
  - `n_text_labels` is the number of text input labels
  - 16 is the number of augmentations
- Preprocessing: apply text prompt on the input text labels

### ImageCLIP

- ailia input shape: (batch_size \* num_segments, 3, 224, 224) RGB channel order
  - `num_segments` is the number of segments the model has been trained on
- Preprocessing: normalization using means [0.48145466, 0.4578275, 0.40821073]
  and standard deviations [0.26862954, 0.26130258, 0.27577711] (RGB channel
  order)

### Fusion Model (Visual Prompt)

- ailia input shape: (batch_size, num_segments, 512)
  - `num_segments` is the number of segments the model has been trained on

## Output

- Zero-Shot Prediction

```bash
### Predicts the top 5 most likely labels among input text labels ###
==============================================================
class_count = 10
+ idx = 0
  category = 2 [driving]
  prob = 0.5602660179138184
+ idx = 1
  category = 3 [driving car]
  prob = 0.3114832639694214
+ idx = 2
  category = 4 [driving truck]
  prob = 0.12353289872407913
+ idx = 3
  category = 9 [talking phone]
  prob = 0.0027049153577536345
+ idx = 4
  category = 7 [reading]
  prob = 0.0006968703237362206

Script finished successfully.
```

### TextCLIP

- ailia Predict API output:
  - `text_features`: encoded text features
    - Shape: (n, 512)
    - Features need to be normalized by the norm over `axis=1` before computing
      the similarity

### ImageCLIP

- ailia Predict API output:
  - `image_features`: encoded image features
    - Shape: (batch_size \* num_segments, 512)

### Fusion Model (Visual Prompt)

- ailia Predict API output:
  - `fused_image_features`: fused image features
    - Shape: (batch_size, 512)
    - Features need to be normalized by the norm over `axis=1` before computing
      the similarity

## Requirements

This model requires additional packages.

```
pip3 install ftfy regex
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

By adding the `--video` option, you can input the video. Webcam input is not supported.
```bash
$ python3 action_clip.py --video VIDEO_PATH
```

You can use the `--text` option if you want to specify custom text input labels.  
```bash
$ python3 action_clip.py --video VIDEO_PATH --text "drinking" --text "eating" --text "laughing"
```

If you want to load custom text input labels from a file, use the `--desc_file` option (1 label/line).
```bash
$ python3 action_clip.py --video VIDEO_PATH --desc_file imagenet_classes.txt
```

## Reference

- [ActionCLIP](https://github.com/sallymmx/ActionCLIP)

## Framework

PyTorch 1.8.1

## Model Format

ONNX opset = 10

## Netron

[vit-32-8f-text_clip.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/action_clip/vit-32-8f-text_clip.onnx.prototxt)  
[vit-32-8f-image_clip.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/action_clip/vit-32-8f-image_clip.onnx.prototxt)  
[vit-32-8f-fusion.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/action_clip/vit-32-8f-fusion.onnx.prototxt)  
