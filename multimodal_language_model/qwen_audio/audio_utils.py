import os
import re
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


def get_T_after_cnn(L_in, dilation=1):
    for padding, kernel_size, stride in eval("[(1,3,1)] + [(1,3,2)] "):
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return L_out


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__), "mel_filters.npz")  # todo
        # os.path.join("assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = N_MELS,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def process_audio(content):
    pattern = r"<audio>(.*?)</audio>"
    audio_urls = re.findall(pattern, content)
    if len(audio_urls) == 0:
        return None

    audios, audio_lens, audio_span_tokens = [], [], []
    for audio_path in audio_urls:
        cache = getattr(process_audio, "cache", {})
        if audio_path in cache:
            mel, audio_len, audio_token_num = cache[audio_path]
            audios.append(mel)
            audio_lens.append(audio_len)
            audio_span_tokens.append(audio_token_num + 2)
            continue

        audio = load_audio(audio_path)
        L = audio.shape[0] if audio.shape[0] <= 480000 else 480000  # max_length < 30s
        mel_len = L // 160
        audio = pad_or_trim(audio.flatten())
        mel = log_mel_spectrogram(audio)
        audio_len_after_cnn = get_T_after_cnn(mel_len)
        audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
        audio_len = [audio_len_after_cnn, audio_token_num]
        audios.append(mel)
        audio_lens.append(audio_len)
        audio_span_tokens.append(audio_token_num + 2)  # add audio bos eos

        cache[audio_path] = (mel, audio_len, audio_token_num)
        process_audio.cache = cache

    input_audio_lengths = np.array(audio_lens)
    input_audios = np.stack(audios, axis=0)

    return {
        "input_audios": input_audios,
        "input_audio_lengths": input_audio_lengths,
        "audio_span_tokens": audio_span_tokens,
        "audio_urls": audio_urls,
    }
