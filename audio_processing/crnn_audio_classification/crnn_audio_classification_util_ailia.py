import numpy
import functools
import ailia
from ailia.audio import spectrogram, mel_scale, complex_norm, get_fb_matrix


class MelspectrogramStretch(object):

    def __init__(self):
        sample_rate = 44100
        num_mels = 128
        fft_length = 2048
        hop_length = fft_length//2
        freq_n = fft_length//2 + 1

        self.stft = functools.partial(spectrogram,
            fft_n=fft_length,
            hop_n=hop_length,
            center_mode=1
        )

        fb = get_fb_matrix(sample_rate,freq_n,mel_n = num_mels, norm=False, htk=True)
        self.msc = functools.partial(mel_scale,
            mel_fb=fb
        )

        # Normalization (pot spec processing)
        self.complex_norm = functools.partial(complex_norm,power=2.)

    def forward(self, data):
        wav,_ = data
        x = self.stft(wav)
        print(x.shape)
        # x -> (fft_length//2+1,channel)

        # print(x.shape)  #([1025, 176, 2])
        x = self.complex_norm(x)
        # print(x.shape)  #([1025, 176])
        x = self.msc(x)
        # print(x.shape)  #([128, 176])

        # Normalize melspectrogram
        # Independent mean, std per batch
        mean = x.mean(keepdims=True)
        std = x.std(keepdims=True)
        x = (x - mean)/std

        lengths = [x.shape[-1]]
        x = x[numpy.newaxis,numpy.newaxis,...] #x.shape = (1,1,128,176)
        return x, lengths
