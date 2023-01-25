from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


class SingleSong:
    # To load one excerpt with arbitrary length, or one full song, for test or validation
    def __init__(self, chunk_len, filter_, hq_path, cutoff, duration=None, start=8):

        hq, sr = read_audio(hq_path)    # high quality target
        lq = lowpass(hq, cutoff, filter_=filter_)  # low quality input

        # CROP
        song_len = lq.shape[-1]

        if duration is None:    # save entire song
            test_start = 0
            test_len = song_len
        else:
            test_start = start * sr    # start from n th second
            test_len = duration * sr
            
        test_len = min(test_len, song_len - test_start)    

        lq = lq[:, test_start:test_start + test_len]
        hq = hq[:, test_start:test_start + test_len]

        self.x_full = lq.copy()
        self.t_full = hq.copy()

        # To have equal length chunks for minibatching
        time_len = lq.shape[-1]
        n_chunks, rem = divmod(time_len, chunk_len)
        lq = lq[..., :-rem or None]    # or None handles rem=0
        hq = hq[..., :-rem or None]    

        # adjust lengths
        self.x_full = self.x_full[..., :lq.shape[-1] or None]
        self.t_full = self.t_full[..., :lq.shape[-1] or None]

        # Save full samples
    
        self.lq = np.split(lq, n_chunks, axis=-1)   # create a lists of chunks
        self.hq = np.split(hq, n_chunks, axis=-1)   # create a lists of chunks

    def get_full_signals(self):
        # Returns full length input and target
        return self.x_full, self.t_full

    def preallocate(self):
        """
        Preallocates the matrix to save all minibatch outputs.
        It is faster to transfer all minibatches from GPU to CPU at once.
        """
        return np.zeros((len(self.lq), *self.lq[0].shape))

    def __len__(self):
        return len(self.lq)

    def __getitem__(self, idx):
        return self.lq[idx], self.hq[idx]


def lowpass(sig, cutoff, filter_=('cheby1', 8), sr=44100):
    """Lowpasses input signal based on a cutoff frequency
    
    Arguments:
        sig {numpy 1d array} -- input signal
        cutoff {int} -- cutoff frequency
    
    Keyword Arguments:
        sr {int} -- sampling rate of the input signal (default: {44100})
        filter_type {str} -- type of filter, only butter and cheby1 are implemented (default: {'butter'})
    
    Returns:
        numpy 1d array -- lowpassed signal
    """
    nyq = sr / 2
    cutoff /= nyq

    if filter_[0] == 'butter':
        B, A = signal.butter(filter_[1], cutoff)
    elif filter_[0] == 'cheby1':
        B, A = signal.cheby1(filter_[1], 0.05, cutoff)
    elif filter_[0] == 'bessel':
        B, A = signal.bessel(filter_[1], cutoff, norm='mag')
    elif filter_[0] == 'ellip':
        B, A = signal.ellip(filter_[1], 0.05, 20, cutoff)

    sig_lp = signal.filtfilt(B, A, sig)
    return sig_lp.astype(np.float32)


def read_audio(path, make_stereo=True):
    sr, audio = wavfile.read(path)
    audio = audio.T
    if np.issubdtype(audio.dtype, np.int16):
        audio = audio.astype(np.float32) / 32768.0
    if len(audio.shape) == 1:    # if mono
        audio = np.expand_dims(audio, axis=0)
        if make_stereo:
            audio = np.repeat(audio, 2, axis=0)
    return audio, sr