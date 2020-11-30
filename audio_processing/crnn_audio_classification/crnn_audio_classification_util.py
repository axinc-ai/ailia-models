import numpy as np
import torch
import torch.nn as nn

from torchaudio.transforms import Spectrogram, MelSpectrogram , ComplexNorm
from torchaudio.transforms import TimeStretch, AmplitudeToDB 
from torch.distributions import Uniform

from torchvision import transforms

class MelspectrogramStretch(object):

    def __init__(self):

        sample_rate=44100
        num_mels=128
        fft_length=2048
        hop_length=fft_length//2

        self.stft = Spectrogram(n_fft=fft_length, win_length=fft_length,
                                       hop_length=None, pad=0, 
                                       power=None, normalized=False)

        self.mst = MelSpectrogram(sample_rate=sample_rate, 
                                                    n_fft=fft_length, 
                                                    hop_length=hop_length, 
                                                    n_mels=num_mels)

        # Normalization (pot spec processing)
        self.complex_norm = ComplexNorm(power=2.)


    def forward(self, data):
        tsf = AudioTransforms()
        sig_t, sr, _ = tsf.apply(data, None)

        length = torch.tensor(sig_t.size(0))
        sr = torch.tensor(sr)
        data = [d.unsqueeze(0).to("cpu") for d in [sig_t, length, sr]]
                            
        # x-> (batch, time, channel)
        x, lengths, _ = data # unpacking seqs, lengths and srs
        # x-> (batch, channel, time)
        xt = x.float().transpose(1,2)
        # xt -> (batch, channel, freq, time)
        x = self.stft(xt)
        # x -> (fft_length//2+1,bins,channel)

        #print(x.shape)  #torch.Size([1, 1, 1025, 173, 2])
        x = self.complex_norm(x)
        #print(x.shape)  #torch.Size([1, 1, 1025, 173])
        x = self.mst.mel_scale(x)
        #print(x.shape)  #torch.Size([1, 1, 128, 173])

        # Normalize melspectrogram
        # Independent mean, std per batch
        non_batch_inds = [1, 2, 3]
        mean = x.mean(non_batch_inds, keepdim=True)
        std = x.std(non_batch_inds, keepdim=True)
        x = (x - mean)/std 

        x = x.to('cpu').detach().numpy().copy()

        lengths = [x.shape[3]]
        return x, lengths

class AudioTransforms(object):

    def apply(self, data, target):
        audio, sr = data
        # audio -> (time, channel)

        # avg
        new_audio= audio.mean(axis=1) if audio.ndim > 1 else audio
        new_audio = new_audio[:,None] 
        new_audio = torch.from_numpy(new_audio)

        return new_audio, sr, target
        

