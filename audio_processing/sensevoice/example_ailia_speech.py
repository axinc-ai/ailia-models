import ailia_speech

import librosa

import os
import urllib.request

# Load target audio
input_file_path = "demo.wav"
if not os.path.exists(input_file_path):
	urllib.request.urlretrieve(
		"https://github.com/axinc-ai/ailia-models/raw/refs/heads/master/audio_processing/whisper/demo.wav",
		"demo.wav"
	)
audio_waveform, sampling_rate = librosa.load(input_file_path, mono = True)

# Infer
speech = ailia_speech.SenseVoice()
speech.initialize_model(model_path = "./models/", model_type = ailia_speech.AILIA_SPEECH_MODEL_TYPE_SENSEVOICE_SMALL)
recognized_text = speech.transcribe(audio_waveform, sampling_rate)
for text in recognized_text:
	print(text)
