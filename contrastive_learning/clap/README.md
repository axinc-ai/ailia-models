# CLAP

Contrastive Language-Audio Pretraining, known as CLAP. Referring to the CLIP architecture, similarly, the CLAP architecture is as follows.

## Input

Audio file
```
24965__www-bonson-ca__bigdogbarking-02.wav
Attribution 3.0 Unported (CC BY 3.0)
https://freesound.org/people/www.bonson.ca/sounds/24965/
```
## Output

Output the cosine similarity between the pre-prepared text embedding and the input audio file embedding. The higher a value of cosine similality is, the closer given text and given audio are in meaning.
```
===== cosine similality between text and audio =====
cossim=0.1514, word=applause applaud clap
cossim=0.2942, word=The crowd is clapping.
cossim=0.0391, word=I love the contrastive learning
cossim=0.0755, word=bell
cossim=-0.0926, word=soccer
cossim=0.0309, word=open the door.
cossim=0.0849, word=applause
cossim=0.4183, word=dog
cossim=0.3819, word=dog barking
  ```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 clap.py
```

If you want to run in onnx mode, you specify `--onnx` option as below.
```bash
$ python3 clap.py --onnx
```

You can run with other wav file by adding `--input` option.
```bash
$ python3 clap.py --input [wav_file]
```

## Reference

[CLAP](https://github.com/LAION-AI/CLAP)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[CLAP_text_text_branch_RobertaModel_roberta-base.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/clap/CLAP_text_text_branch_RobertaModel_roberta-base.onnx.prototxt)  
[CLAP_text_projection_LAION-Audio-630K_with_fusion.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/clap/CLAP_text_projection_LAION-Audio-630K_with_fusion.onnx.prototxt)  
[CLAP_audio_LAION-Audio-630K_with_fusion.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/clap/CLAP_audio_LAION-Audio-630K_with_fusion.onnx.prototxt)
