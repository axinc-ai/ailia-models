#from tensorflow.contrib.training import HParams
from glob import glob
import os, pickle

class DictDotNotation(dict): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.__dict__ = self 

# Default hyperparameters
hparams = DictDotNotation({
	"num_mels":80,  # Number of mel-spectrogram channels and local conditioning dimensionality
	#  network
	"rescale":True,  # Whether to rescale audio prior to preprocessing
	"rescaling_max":0.9,  # Rescaling value

	# For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, 
	# also consider clipping your samples to smaller chunks)
	"max_mel_frames":900,
	# Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3
	#  and still getting OOM errors.
	
	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	# Does not work if n_ffit is not multiple of hop_size!!
	"use_lws":False,
	
	"n_fft":800,  # Extra window size is filled with 0 paddings to match this parameter
	"hop_size":200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
	"win_size":800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
	"sample_rate":16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
	
	"frame_shift_ms":None,  # Can replace hop_size parameter. (Recommended: 12.5)
	
	# Mel and Linear spectrograms normalization/scaling and clipping
	"signal_normalization":True,
	# Whether to normalize mel spectrograms to some predefined range (following below parameters)
	"allow_clipping_in_normalization":True,  # Only relevant if mel_normalization = True
	"symmetric_mels":True,
	# Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
	# faster and cleaner convergence)
	"max_abs_value":4.,
	# max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
	# be too big to avoid gradient explosion, 
	# not too small for fast convergence)
	"normalize_for_wavenet":True,
	# whether to rescale to [0, 1] for wavenet. (better audio quality)
	"clip_for_wavenet":True,
	# whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)
	
	# Contribution by @begeekmyfriend
	# Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
	# levels. Also allows for better G&L phase reconstruction)
	"preemphasize":True,  # whether to apply filter
	"preemphasis":0.97,  # filter coefficient.
	
	# Limits
	"min_level_db":-100,
	"ref_level_db":20,
	"fmin":55,
	# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
	# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	"fmax":7600,  # To be increased/reduced depending on data.
	
	# Griffin Lim
	"power":1.5,
	# Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
	"griffin_lim_iters":60,
	# Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
})


def hparams_debug_string():
	values = hparams.values()
	hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
	return "Hyperparameters:\n" + "\n".join(hp)
