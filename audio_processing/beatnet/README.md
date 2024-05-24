# BeatNet: CRNN and Particle Filtering for Online Joint Beat Downbeat and Meter Tracking
## Input

* **Audio path**

Path to the audio file you want to detect the beats of.
The example in this directory is a drum sound repeated at 120 beats per minute (bpm).

## Output

* **Predicted beats**

The output will be an array with shape (num_beats, 2).
The first column represents the time in seconds in which the beat appeared, and the second column represents the beat number.
The downbeat (the first beat of the bar) is represented by the number 1, and the other beats are numbered sequentially



## Usage
An Internet connection is required when running the script for the first time, as the model files will be downloaded automatically.

The model will predict the beats in the input audio using CRNN (the model in .onnx format) and other inference models (PF and DBN).
There are 3 modes that you can choose from. The default is offline mode with DBN as the inference model.

#### Offline mode
```bash
$ python3 beatnet.py --mode offline --inference-model DBN -i input.mp3
```
**output**
```bash
[[0.48 1.  ]
 [1.   2.  ]
 [1.5  3.  ]
 ...
 [9.5  3.  ]
 [9.98 4.  ]]
```
Offline mode can only be used with DBN (Dense Bayesian Networks).
While using this mode, the entire audio file is available for the model to use at the time of inference.

#### Online mode
```bash
$ python3 beatnet.py --mode online --inference-model PF -i input.mp3
```
Online mode accepts both DBN and PF (Particle Filter) as its inference models.
While using this mode, only the preceeding sections of the audio file is available at the time of inference.

#### Realtime mode
```bash
$ python3 beatnet.py --mode realtime --inference-model DBN -i input.mp3
```
**output**
```bash
*beat!
beat!
...
*beat!
beat!
[[0.48 1.  ]
 [0.98 2.  ]
 [1.48 1.  ]
 ...
 [9.48 1.  ]
 [9.98 2.  ]]
```
Realtime mode is similar to online mode, but it processes and shows the result in real time.

#### Switching the model weights
```bash
$ python3 beatnet.py --weight 2
```
You can switch the weights of the CRNN model using the --weight argument. Use 1 for weights trained with GTZAN, 2 for weights trained with Ballroom, and 3 for weights trained with Rock_corpus. The default is 1.

## Reference

* [BeatNet](https://github.com/mjhydri/BeatNet)

## Framework

PyTorch


## Model Format

ONNX opset=11

## Netron

- [beatnet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/beatnet/beatnet.onnx.prototxt)