import sys
import numpy as np
import ailia.audio as ailia_audio


# ======================
# Pre/Post process
# ======================
def preemphasis(data, coeff=0.97):

    return ailia_audio.linerfilter(np.array([1,-coeff]), np.array([1]), data).astype(np.float32)


def inv_preemphasis(data, coeff=0.97):

    return ailia_audio.linerfilter(np.array([1]), np.array([1,-coeff]), data).astype(np.float32)


def lowpass(data, stop_freq, sample_freq, N=4):
    if( stop_freq == 10000 and sample_freq == 22050) or ( stop_freq == 20000 and sample_freq == 44100):
        b = np.array([0.68166451, 2.72665802, 4.08998703, 2.72665802, 0.68166451])
        a = np.array([1.        , 3.238043,   3.99120175, 2.21272074, 0.4646666 ])
    else:
        raise ValueError('illegal sample freqency.')
    data = ailia_audio.filterfilter(b,a, data)

    return data


def tfconvert(x, window_len, hop_len, mult, window='hann') :
    y = ailia_audio.spectrogram(x, fft_n=window_len, hop_n=hop_len, center_mode=2, norm_type="scipy",win_type=window)

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
