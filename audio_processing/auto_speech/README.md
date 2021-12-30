# AutoSpeech

## Input

Audio file
```
Wav file from The VoxCeleb1 Dataset https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html

Default input: wav/id10283/oGZsanLiXsY/00004.wav
```

Please download the test data set (https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip) to check various data.

## Output

- Identification mode  
  Top 5 label.
  ```
  Top5: id10283, id11084, id10200, id11064, id10404
  ```

- Verification mode  
  Degree of similarity.
  ```
  similar: 0.42575997
  verification: match (threshold: 0.260)
  ```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 auto_speech.py
```
It outputs top 5 label. (identification mode)

If you want to specify the input file, put the path after the `--input` option.
```bash
$ python3 auto_speech.py --input wav/id10283/oGZsanLiXsY/00004.wav
```

When two files are specified with the `--input1` and `--input2` options,
check if two audio files belong to the same person. (verification mode)
```bash
$ python3 auto_speech.py --input1 wav/id10270/8jEAjG6SegY/00008.wav --input2 wav/id10270/x6uYqmx31kE/00001.wav
```

## Reference

[AutoSpeech: Neural Architecture Search for Speaker Recognition](https://github.com/VITA-Group/AutoSpeech)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[proposed_iden.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/auto_speech/proposed_iden.onnx.prototxt)  
[proposed_classifier.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/auto_speech/proposed_classifier.onnx.prototxt)  
[proposed_veri.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/auto_speech/proposed_veri.onnx.prototxt)
