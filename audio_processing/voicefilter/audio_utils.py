# adapted from Keith Ito's tacotron implementation
# https://github.com/keithito/tacotron/blob/master/util/audio.py

import librosa
import numpy as np


class Dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


hp = Dotdict({
    "audio": {
        "n_fft": 1200,
        "num_freq": 601,  # n_fft//2 + 1
        "sample_rate": 16000,
        "hop_length": 160,
        "win_length": 400,
        "min_level_db": -100.0,
        "ref_level_db": 20.0,
    },
    "embedder": {
        "n_fft": 512,
        "num_mels": 40,
    }
})


class Audio:
    def __init__(self):
        self.mel_basis = librosa.filters.mel(
            sr=hp.audio.sample_rate,
            n_fft=hp.embedder.n_fft,
            n_mels=hp.embedder.num_mels)

    def get_mel(self, y):
        y = librosa.core.stft(
            y=y, n_fft=hp.embedder.n_fft,
            hop_length=hp.audio.hop_length,
            win_length=hp.audio.win_length,
            window='hann')
        magnitudes = np.abs(y) ** 2
        mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)

        return mel

    def wav2spec(self, y):
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - hp.audio.ref_level_db
        S, D = self.normalize(S), np.angle(D)
        S, D = S.T, D.T  # to make [time, freq]

        return S, D

    def spec2wav(self, spectrogram, phase):
        spectrogram, phase = spectrogram.T, phase.T
        # used during inference only
        # spectrogram: enhanced output
        # phase: use noisy input's phase, so no GLA is required
        S = self.db_to_amp(self.denormalize(spectrogram) + hp.audio.ref_level_db)

        return self.istft(S, phase)

    def stft(self, y):
        return librosa.stft(
            y=y, n_fft=hp.audio.n_fft,
            hop_length=hp.audio.hop_length,
            win_length=hp.audio.win_length)

    def istft(self, mag, phase):
        stft_matrix = mag * np.exp(1j * phase)
        return librosa.istft(
            stft_matrix,
            hop_length=hp.audio.hop_length,
            win_length=hp.audio.win_length)

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))

    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def normalize(self, S):
        return np.clip(S / -hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * -hp.audio.min_level_db
