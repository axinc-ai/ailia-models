import numpy as np
from scipy.ndimage import zoom

import librosa
from typing import Optional

def power_to_db(spectrogram, reference, min_value, db_range=None):
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")

    reference = max(min_value, reference)

    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
    spectrogram = 10.0 * (np.log10(spectrogram) - np.log10(reference))

    if db_range is not None:
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        spectrogram = np.clip(spectrogram, a_min=spectrogram.max() - db_range, a_max=None)

    return spectrogram
def _np_extract_fbank_features(waveform: np.array) -> np.ndarray:
    """
    Compute the log-mel spectrogram of the provided `waveform` using the Hann window. In CLAP, two different filter
    banks are used depending on the truncation pattern:
        - `self.mel_filters`: they correspond to the default parameters of `torchaudio` which can be obtained from
          calling `torchaudio.transforms.MelSpectrogram().mel_scale.fb`. These filters are used when `truncation`
          is set to `"fusion"`.
        - `self.mel_filteres_slaney` : they correspond to the default parameters of `librosa` which used
          `librosa.filters.mel` when computing the mel spectrogram. These filters were only used in the original
          implementation when the truncation mode is not `"fusion"`.
    """
    stfted = librosa.stft(
        waveform,
        n_fft = 1024,
        hop_length = 480, 
        win_length = 1024,
        window = 'hann',
        center = True,
        pad_mode = 'reflect'
    )
    spectrogram = np.abs(stfted, dtype=np.float64) ** 2
    
    mel_basis = librosa.filters.mel(
        sr=48000,
        n_fft=1024,
        n_mels=64,
        fmin=50,
        fmax=14000,
        htk=True,
        norm=None
    )
    spectrogram = np.maximum(1e-10, np.dot(mel_basis, spectrogram))

    log_mel_spectrogram = power_to_db(
        spectrogram,
        reference=1.0,
        min_value=1e-10,
        db_range=None
    )
    return log_mel_spectrogram.T

def resize_array(array, size):
    """
    Resize a 2D array to the new shape using interpolation.
    
    Parameters:
    - array: np.ndarray, the input array to be resized.
    - size: tuple, the desired shape of the output array.os
    
    Returns:
    - np.ndarray, the resized array.
    """
    zoom_factors = [n / o for n, o in zip(size, array.shape)]
    resized_array = zoom(array, zoom_factors, order=1)  # order=1 for bilinear interpolation
    return resized_array

def _random_mel_fusion(mel, total_frames, chunk_frames):
    ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
    if len(ranges[1]) == 0:
        # if the audio is too short, we just use the first chunk
        ranges[1] = [0]
    if len(ranges[2]) == 0:
        # if the audio is too short, we just use the first chunk
        ranges[2] = [0]
    # randomly choose index for each part
    idx_front = np.random.choice(ranges[0])
    idx_middle = np.random.choice(ranges[1])
    idx_back = np.random.choice(ranges[2])
    mel_chunk_front = mel[idx_front : idx_front + chunk_frames, :]
    mel_chunk_middle = mel[idx_middle : idx_middle + chunk_frames, :]
    mel_chunk_back = mel[idx_back : idx_back + chunk_frames, :]

    #mel = torch.tensor(mel[None, None, :])
    #mel_shrink = torch.nn.functional.interpolate(
    #    mel, size=[chunk_frames, 64], mode="bilinear", align_corners=False
    #)
    #mel_shrink = mel_shrink[0][0].numpy()
    mel_shrink = resize_array(mel, size=[chunk_frames, 64])

    mel_fusion = np.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], axis=0)
    return mel_fusion

def _get_input_mel(waveform: np.array, max_length: int = 480000) -> np.array:
       """
       Extracts the mel spectrogram and prepares it for the mode based on the `truncation` and `padding` arguments.
       Four different path are possible:
           - `truncation="fusion"` and the length of the waveform is greater than the max length: the mel spectrogram
             will be computed on the entire audio. 3 random crops and a dowsampled version of the full mel spectrogram
             are then stacked together. They will later be used for `feature_fusion`.
           - `truncation="rand_trunc"` and the length of the waveform is smaller than the max length: the audio is
             padded based on `padding`.
           - `truncation="fusion"` and the length of the waveform is smaller than the max length: the audio is padded
             based on `padding`, and is repeated `4` times.
           - `truncation="rand_trunc"` and the length of the waveform is greater than the max length: the mel
             spectrogram will be computed on a random crop of the waveform.
       """
       if waveform.shape[0] > max_length:
            mel = _np_extract_fbank_features(waveform)
            chunk_frames = max_length // 480 + 1  # the +1 related to how the spectrogram is computed
            total_frames = mel.shape[0]
            if chunk_frames == total_frames:
                # there is a corner case where the audio length is larger than max_length but smaller than max_length+hop_length.
                # In this case, we just use the whole audio.
                input_mel = np.stack([mel, mel, mel, mel], axis=0)
                longer = False
            else:
                input_mel = _random_mel_fusion(mel, total_frames, chunk_frames)
                longer = True
       else:
            longer = False
            # only use repeat as a new possible value for padding. you repeat the audio before applying the usual max_length padding
            if waveform.shape[0] < max_length:
                n_repeat = int(max_length / len(waveform))
                waveform = np.tile(waveform, n_repeat)
                waveform = np.pad(waveform, (0, max_length - waveform.shape[0]), mode="constant", constant_values=0)
            input_mel = _np_extract_fbank_features(waveform)
            input_mel = np.stack([input_mel, input_mel, input_mel, input_mel], axis=0)
       return input_mel, longer

def feature_extractor(raw_speech):
    input_mel, _ = _get_input_mel(raw_speech)
    return input_mel, True