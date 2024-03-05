import numpy as np
import librosa
from skimage.transform import resize


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def get_mel(audio_data, audio_cfg):
    """
    # mel shape: (n_mels, T)
    mel_torch = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg['sample_rate'],
        n_fft=audio_cfg['window_size'],
        win_length=audio_cfg['window_size'],
        hop_length=audio_cfg['hop_size'],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=64,
        f_min=audio_cfg['fmin'],
        f_max=audio_cfg['fmax']
    )(audio_data)

    # we use log mel spectrogram as input
    mel_torch = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel_torch)
    mel_torch = mel_torch.T # (T, n_mels)
    mel_torch = mel_torch.to('cpu').detach().numpy().copy()
    """

    # Align to librosa:
    mel_librosa = librosa.feature.melspectrogram(
        y=audio_data,
        sr=audio_cfg['sample_rate'],
        n_fft=audio_cfg['window_size'],
        hop_length=audio_cfg['hop_size'],
        win_length=audio_cfg['window_size'],
        center=True,
        pad_mode="reflect",
        power=2.0,
        n_mels=64,
        norm=None,
        htk=True,
        fmin=audio_cfg['fmin'],
        fmax=audio_cfg['fmax']
    )
    mel_librosa = librosa.amplitude_to_db(mel_librosa, top_db=None)
    mel_librosa = mel_librosa.transpose(1, 0)

    return mel_librosa


def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg):
    """
    Calculate and add audio features to sample.
    Sample: a dict containing all the data of current sample.
    audio_data: a tensor of shape (T) containing audio data.
    max_len: the maximum length of audio data.
    data_truncating: the method of truncating data.
    data_filling: the method of filling data.
    audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
    """
    if len(audio_data) > max_len:
        if data_truncating == "fusion":
            # fusion
            mel = get_mel(audio_data, audio_cfg)
            # split to three parts
            chunk_frames = max_len // audio_cfg['hop_size']+1  # the +1 related to how the spectrogram is computed
            total_frames = mel.shape[0]
            if chunk_frames == total_frames:
                # there is a corner case where the audio length is
                # larger than max_len but smaller than max_len+hop_size.
                # In this case, we just use the whole audio.
                mel_fusion = np.stack([mel, mel, mel, mel], axis=0)
                longer = [[False]]
            else:
                ranges = np.array_split(list(range(0, total_frames-chunk_frames+1)), 3)
                # print('total_frames-chunk_frames:', total_frames-chunk_frames,
                #       'len(audio_data):', len(audio_data),
                #       'chunk_frames:', chunk_frames,
                #       'total_frames:', total_frames)
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
                # select mel
                mel_chunk_front = mel[idx_front:idx_front+chunk_frames, :]
                mel_chunk_middle = mel[idx_middle:idx_middle+chunk_frames, :]
                mel_chunk_back = mel[idx_back:idx_back+chunk_frames, :]

                # shrink the mel
                # Output may differ between torchvision.transforms.Resize and skimage.transform.resize.
                #mel_shrink_torch = torch.from_numpy(mel[None])
                #mel_shrink_torch = torchvision.transforms.Resize(size=[chunk_frames, 64])(mel_shrink_torch)[0]
                #mel_shrink_torch = mel_shrink_torch.to('cpu').detach().numpy().copy()
                mel_shrink_numpy = resize(mel, (chunk_frames, 64), preserve_range=True, anti_aliasing=True, mode='edge')
                # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                # stack
                mel_fusion = np.stack([mel_chunk_front, mel_chunk_middle, mel_chunk_back, mel_shrink_numpy], axis=0)
                longer = [[True]]
        # random crop to max_len (for compatibility)
        overflow = len(audio_data) - max_len
        idx = np.random.randint(0, overflow + 1)
        audio_data = audio_data[idx: idx + max_len]

    else:  # padding if too short
        if len(audio_data) < max_len:  # do nothing if equal
            if data_filling == "repeatpad":
                n_repeat = int(max_len/len(audio_data))
                audio_data = np.tile(audio_data, n_repeat)
                # audio_data = audio_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                # audio_data = F.interpolate(audio_data,size=max_len,mode="bicubic")[0,0,0]
                audio_data = np.pad(audio_data, [(0, max_len - len(audio_data))], "constant")
            elif data_filling == "pad":
                audio_data = np.pad(audio_data, [(0, max_len - len(audio_data))], "constant")
            elif data_filling == "repeat":
                n_repeat = int(max_len/len(audio_data))
                audio_data = np.tile(audio_data, n_repeat+1)[:max_len]
                
        if data_truncating == 'fusion':
            mel = get_mel(audio_data, audio_cfg)
            mel_fusion = np.stack([mel, mel, mel, mel], axis=0)
        longer = [[False]]

    return longer, mel_fusion, audio_data