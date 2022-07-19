import numpy
import functools
import ailia
from ailia.audio import spectrogram, mel_scale, complex_norm, get_fb_matrix, mel_spectrogram


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

        # Fused api
        self.mel_spectrogram = functools.partial(mel_spectrogram, sample_rate=sample_rate,
                        fft_n = fft_length,
                        hop_n =hop_length,
                        win_n = fft_length,
                        win_type = 1,
                        center_mode = 1,
                        power = 2.,
                        fft_norm_type = 0,
                        f_min = 0.0,
                        f_max = sample_rate/2,
                        mel_n = num_mels,
                        mel_norm = 0,
                        htk = True)

    def forward(self, data):
        wav,_ = data

        # Stereo to monoral
        if len(wav.shape)>=2:
            wav = numpy.mean(wav, axis=1)

        if True:
            # Fused API
            x = self.mel_spectrogram(wav)
        else:
            # Independent API

            x = self.stft(wav)
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
