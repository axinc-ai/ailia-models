import numpy as np
import torch
import torch.nn as nn

from torchaudio.transforms import Spectrogram, MelSpectrogram , ComplexNorm
from torchaudio.transforms import TimeStretch, AmplitudeToDB 
from torch.distributions import Uniform

from torchvision import transforms

def _num_stft_bins(lengths, fft_length, hop_length, pad):
    return (lengths + 2 * pad - fft_length + hop_length) // hop_length

class MelspectrogramStretch(MelSpectrogram):

    def __init__(self,):

        hop_length=None
        sample_rate=44100
        num_mels=128
        fft_length=2048
        stretch_param=[0.4, 0.4]

        super(MelspectrogramStretch, self).__init__(sample_rate=sample_rate, 
                                                    n_fft=fft_length, 
                                                    hop_length=hop_length, 
                                                    n_mels=num_mels)

        self.stft = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length, pad=self.pad, 
                                       power=None, normalized=False)
        
        # Normalization (pot spec processing)
        self.complex_norm = ComplexNorm(power=2.)

    def forward(self, x, lengths=None):
        x = self.stft(x)

        if lengths is not None:
            lengths = _num_stft_bins(lengths, self.n_fft, self.hop_length, self.n_fft//2)
            lengths = lengths.long()
        
        x = self.complex_norm(x)
        x = self.mel_scale(x)

        # Normalize melspectrogram
        # Independent mean, std per batch
        non_batch_inds = [1, 2, 3]
        mean = x.mean(non_batch_inds, keepdim=True)
        std = x.std(non_batch_inds, keepdim=True)
        x = (x - mean)/std 

        if lengths is not None:
            return x, lengths
        return x

class AudioTransforms(object):

    def apply(self, data, target):
        audio, sr = data
        # audio -> (time, channel)

        # avg
        new_audio= audio.mean(axis=1) if audio.ndim > 1 else audio
        new_audio = new_audio[:,None] 
        new_audio = torch.from_numpy(new_audio)

        return new_audio, sr, target
        

