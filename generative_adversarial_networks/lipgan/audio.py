import librosa
import numpy as np
from scipy import signal

min_level_db = -100
ref_level_db = 20
max_abs_value = 4.
n_fft = 800
hop_size = 200
win_size = 800
sample_rate = 16000
num_mels = 80
fmin = 55
fmax = 7600

def preemphasis(wav):
    # new_wav[t] = 1 * wav[t] - 0.97 * wav[t-1]
    k = 0.97
    return signal.lfilter([1, -k], [1], wav)

def _amp_to_db(x):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _normalize(S):
    return np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
                    -max_abs_value, max_abs_value)

def melspectrogram(wav, ailia_audio):
    wav = preemphasis(wav)
    if ailia_audio:
        import ailia.audio
        D = ailia.audio.mel_spectrogram(wav, sample_rate = sample_rate, fft_n = n_fft, hop_n = hop_size, win_type ="hann", center_mode = 1, power = 1.0,
                                        fft_norm_type = 0, f_min=fmin, f_max=fmax, mel_n = num_mels, mel_norm=1, htk=False)
    else:
        D = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_size, win_length=win_size) # default args : window="hann", center=True, pad_mode="reflect"
        D = np.abs(D)
        _mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax) # default atgs : htk=False, norm="slaney"
        D = np.dot(_mel_basis, D)
    S = _amp_to_db(D) - ref_level_db
    return _normalize(S)
