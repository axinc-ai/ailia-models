# Microsoft CLAP

## Input

**audio file**

Audio file in wav format to use as the model's input. 
Default file name is [input.wav](./input.wav)
(source: https://freesound.org/people/InspectorJ/sounds/456440/)



**text file**

A text file containing sentences separated by new lines.
Default file name is [captions.txt](./captions.txt)

## Output

**Cosine similarities**

Cosine similarity between the input audio and the sentences in the text file.

## Usage
Internet connection is required when running the script for the first time, as the model files will be automatically downloaded.

Running the script will compute the cosine similarities between the audio and the captions, using audio and language encoder models train by contrastive training.

You can switch the versions of the encoder model's weight (2022 or 2023) by specifying the version using the argument ```-v``` or ```--version```.
For more information on arguments, try running ```python3 msclap.py --help```
```bash
$ python3 msclap.py -t captions.txt -a input.wav -v 2023
 INFO arg_utils.py (13) : Start!
 INFO arg_utils.py (158) : env_id updated to 0
 INFO arg_utils.py (163) : env_id: 0
 INFO arg_utils.py (166) : CPU
 INFO msclap.py (167) : input_text: ['Dog barking.', 'Birds whistling.', 'Car passing by.', 'Wind blowing.', 'Water flowing.', 'People talking.']
 INFO msclap.py (170) : inference has started...
Similarity: 
    Birds whistling.: 0.41247469186782837
       Wind blowing.: 0.2643369734287262
      Water flowing.: 0.23884761333465576
     Car passing by.: 0.22803542017936707
     People talking.: 0.17387858033180237
        Dog barking.: 0.11309497803449631
 INFO msclap.py (192) : Script finished successfully.
```

## Reference

* [CLAP](https://github.com/microsoft/CLAP)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[caption_model_2023.onnx.prototxt]()

[audio_model_2023.onnx.prototxt]()

[caption_model_2022.onnx.prototxt]()

[audio_model_2022.onnx.prototxt]()



