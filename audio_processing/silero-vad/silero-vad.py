﻿import os
import sys
import time

import numpy as np
import librosa
import soundfile as sf

import ailia
from pprint import pprint

from utils_vad import (get_speech_timestamps,
                       save_audio,
                       read_audio,
                       VADIterator,
                       collect_chunks,
                       OnnxWrapper)

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'silero_vad.onnx'
MODEL_PATH = 'silero_vad.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/silero-vad/'

WAVE_PATH = "en_example.wav"
SAVE_PATH = 'only_speech.wav'

# Audio
SAMPLING_RATE = 16000

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Silero VAD', WAVE_PATH, SAVE_PATH, input_ftype='audio'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)

# ======================
# Logic
# ======================

def audio_recognition(model):
  # **Speech timestapms from full audio**

  wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)
  # get speech timestamps from full audio file
  speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
  logger.info("Speech timestamp")
  logger.info(speech_timestamps)

  # merge all speech chunks to one audio
  save_audio('only_speech.wav',
            collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE) 

  ## using VADIterator class
  vad_iterator = VADIterator(model)
  wav = read_audio(f'en_example.wav', sampling_rate=SAMPLING_RATE)

  logger.info("VADIterator")
  window_size_samples = 1536 # number of samples in a single audio chunk
  for i in range(0, len(wav), window_size_samples):
      chunk = wav[i: i+ window_size_samples]
      if len(chunk) < window_size_samples:
        break
      speech_dict = vad_iterator(chunk, return_seconds=True)
      if speech_dict:
          logger.info(speech_dict)
  vad_iterator.reset_states() # reset model states after each audio

  ## just probabilities
  logger.info("Speech Probablities")
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

  logger.info(speech_probs[:10]) # first 10 chunks predicts

  logger.info("Script finish successfully.")


# ======================
# Main
# ======================

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    
    # disable FP16
    if "FP16" in ailia.get_environment(args.env_id).props or sys.platform == 'Darwin':
        logger.warning('This model do not work on FP16. So use CPU mode.')
        args.env_id = 0
        
    if not args.onnx:
        env_id = args.env_id
        session = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime
        session = onnxruntime.InferenceSession(WEIGHT_PATH)

    model = OnnxWrapper('silero_vad.onnx')
    model.session = session
    model.ailia = not args.onnx

    audio_recognition(model)


if __name__ == '__main__':
    main()
