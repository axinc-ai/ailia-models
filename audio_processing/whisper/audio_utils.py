import numpy as np
import librosa
import ffmpeg

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = (N_SAMPLES // HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input


def load_audio(file: str, sr: int = SAMPLE_RATE):
    out, _ = (
        ffmpeg.input(file, threads=0)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
        .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
    )
    out = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    return out


def pad_or_trim(array, length=N_SAMPLES, axis=-1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array


def mel_filters(n_mels=N_MELS):
    """
    the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    """
    filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)

    return filters


def log_mel_spectrogram(audio, n_mels=N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: np.ndarray
    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    A Tensor that contains the Mel spectrogram, shape = (80, n_frames)
    """
    stft = librosa.stft(
        y=audio, n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        pad_mode="reflect",
    )
    magnitudes = np.abs(stft[:, :-1]) ** 2

    filters = mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec
