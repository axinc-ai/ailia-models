import sys
import numpy as np
from scipy import signal


# ======================
# Pre/Post process
# ======================
def preemphasis(data, coeff=0.97):

    return signal.lfilter([1,-coeff], [1], data).astype(np.float32)


def inv_preemphasis(data, coeff=0.97):

    return signal.lfilter([1], [1,-coeff], data).astype(np.float32)


def lowpass(data, stop_freq, sample_freq, N=4):
    wn = 2.0 * stop_freq / sample_freq
    b, a = signal.butter(N, wn, btype="low")
    data = signal.filtfilt(b,a, data)

    return data


def tfconvert(x, window_len, hop_len, mult, window='hann') :
    noverlap = window_len - hop_len
    _, _, y = signal.stft(x, window=window, nperseg=window_len, noverlap=noverlap)

    y_re = np.real(y) * (window_len//2 + 1)
    y_im = np.imag(y) * (window_len//2 + 1)

    y_mag = np.log(np.sqrt(y_re ** 2 + y_im ** 2)+1.0).astype(np.float32)
    y_phase = np.arctan2(y_im, y_re).astype(np.float32)

    y_mag = zero_pad(y_mag, mult)
    y_phase = zero_pad(y_phase, mult)

    return y_mag, y_phase


def zero_pad(x, mult) :
    mod = x.shape[2] % mult
    if mod > 0 :
        pad = mult - mod
        x = np.concatenate(( x, np.zeros((x.shape[0], x.shape[1], pad), dtype=np.float32) ), axis=2)
    return x


def calc_time(sample_len ,sr) :
    quot = sample_len // sr
    rem = (sample_len % sr) / sr
    min = quot // 60
    sec = quot % 60 + rem
    print('Time length : {}min {:.02f}sec'.format(min,sec))
