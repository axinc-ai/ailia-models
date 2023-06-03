dependencies = ['torch', 'torchaudio']
import torch
import json
import os
from utils_vad import (get_speech_timestamps,
                       save_audio,
                       read_audio,
                       VADIterator,
                       collect_chunks,
                       OnnxWrapper)

# test audio
# https://models.silero.ai/vad_models/en.wav

def silero_vad(onnx=False, force_onnx_cpu=False):
    model_dir = os.path.join(os.path.dirname(__file__), 'files')
    model = OnnxWrapper(os.path.join(model_dir, 'silero_vad.onnx'), force_onnx_cpu)
    utils = (get_speech_timestamps,
             save_audio,
             read_audio,
             VADIterator,
             collect_chunks)

    return model, utils

SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)

#from IPython.display import Audio
from pprint import pprint

# %%
USE_ONNX = True # change this to True if you want to test onnx model
  
model, utils = silero_vad(onnx = True)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

# **Speech timestapms from full audio**

# %%
wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)
# get speech timestamps from full audio file
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
pprint(speech_timestamps)

# %%
# merge all speech chunks to one audio
save_audio('only_speech.wav',
           collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE) 
#Audio('only_speech.wav')

# %% [markdown]
# ## Stream imitation example

# %%
## using VADIterator class

vad_iterator = VADIterator(model)
wav = read_audio(f'en_example.wav', sampling_rate=SAMPLING_RATE)

window_size_samples = 1536 # number of samples in a single audio chunk
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+ window_size_samples]
    if len(chunk) < window_size_samples:
      break
    speech_dict = vad_iterator(chunk, return_seconds=True)
    if speech_dict:
        print(speech_dict, end=' ')
vad_iterator.reset_states() # reset model states after each audio

# %%
## just probabilities

wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)
speech_probs = []
window_size_samples = 1536
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+ window_size_samples]
    if len(chunk) < window_size_samples:
      break
    speech_prob = model(chunk, SAMPLING_RATE).item()
    speech_probs.append(speech_prob)
vad_iterator.reset_states() # reset model states after each audio

print(speech_probs[:10]) # first 10 chunks predicts


