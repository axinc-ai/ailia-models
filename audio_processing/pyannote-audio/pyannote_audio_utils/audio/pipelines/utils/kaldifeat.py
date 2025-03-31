# https://github.com/yuyq96/kaldifeat

import numpy as np
import csv

# ---------- デバック ----------
def save_array_to_csv(filename, array):
    # 配列が2次元でない場合、2次元に変換する
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(array)


# ---------- feature-window ----------

def sliding_window(x, window_size, window_shift):
    shape = x.shape[:-1] + (x.shape[-1] - window_size + 1, window_size)
    strides = x.strides + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)[::window_shift]


def func_num_frames(num_samples, window_size, window_shift, snip_edges):
    if snip_edges:
        if num_samples < window_size:
            return 0
        else:
            return 1 + ((num_samples - window_size) // window_shift)
    else:
        return (num_samples + (window_shift // 2)) // window_shift


def func_dither(waveform, dither_value):
    if dither_value == 0.0:
        return waveform
    waveform += np.random.normal(size=waveform.shape).astype(waveform.dtype) * dither_value
    return waveform


def func_remove_dc_offset(waveform):
    return waveform - np.mean(waveform)


def func_log_energy(waveform):
    return np.log(np.dot(waveform, waveform).clip(min=np.finfo(waveform.dtype).eps))


def func_preemphasis(waveform, preemph_coeff):
    if preemph_coeff == 0.0:
        return waveform
    assert 0 < preemph_coeff <= 1
    waveform[1:] -= preemph_coeff * waveform[:-1]
    waveform[0] -= preemph_coeff * waveform[0]
    # print(waveform.shape) #(400,)
    return waveform


def sine(M):
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, float)
    n = np.arange(0, M)
    return np.sin(np.pi * n / (M - 1))


def povey(M):
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, float)
    n = np.arange(0, M)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1))) ** 0.85


def feature_window_function(window_type, window_size, blackman_coeff):
    assert window_size > 0
    if window_type == 'hanning':
        return np.hanning(window_size)
    elif window_type == 'sine':
        return sine(window_size)
    elif window_type == 'hamming':
        return np.hamming(window_size)
    elif window_type == 'povey':
        return povey(window_size)
    elif window_type == 'rectangular':
        return np.ones(window_size)
    elif window_type == 'blackman':
        window_func = np.blackman(window_size)
        if blackman_coeff == 0.42:
            return window_func
        else:
            return window_func - 0.42 + blackman_coeff
    else:
        raise ValueError('Invalid window type {}'.format(window_type))


def process_window(window, dither, remove_dc_offset, preemphasis_coefficient, window_function, raw_energy):
    
    # print(window[0])
    
    if dither != 0.0:
        window = func_dither(window, dither)
    if remove_dc_offset:
        window = func_remove_dc_offset(window)
    if raw_energy:
        log_energy = func_log_energy(window)
    if preemphasis_coefficient != 0.0:
        window = func_preemphasis(window, preemphasis_coefficient)
    window *= window_function
    # print(window.shape)
    if not raw_energy:
        log_energy = func_log_energy(window)
    return window, log_energy


def extract_window(waveform, blackman_coeff, dither, window_size, window_shift,
                   preemphasis_coefficient, raw_energy, remove_dc_offset,
                   snip_edges, window_type, dtype):
    num_samples = len(waveform)
    num_frames = func_num_frames(num_samples, window_size, window_shift, snip_edges)
    num_samples_ = (num_frames - 1) * window_shift + window_size
    
    # print("window_size {}".format(window_size)) #400
    # print("window_shift {}".format(window_shift)) #160
    # print("snip_edges {}".format(snip_edges)) #True
    
    # print("num_samples {}".format(num_samples)) #160000
    # print("num_frames {}".format(num_frames)) #998 dummyのときは98
    # print("num_samples_ {}".format(num_samples_)) #159920
    
    if snip_edges:
        waveform = waveform[:num_samples_]
    else:
        offset = window_shift // 2 - window_size // 2
        waveform = np.concatenate([
            waveform[-offset - 1::-1],
            waveform,
            waveform[:-(offset + num_samples_ - num_samples + 1):-1]
        ])
        
    # print(waveform.shape) #159920
    
    frames = sliding_window(waveform, window_size=window_size, window_shift=window_shift)
    # print(frames.shape) #(998, 400)
    # if(waveform[0] == 32768):
    #     print(frames) #[     32768      65536      98304 ...   13041664...
        
    frames = frames.astype(dtype)
    log_enery = np.empty(frames.shape[0], dtype=dtype)
    # for i in range(frames.shape[0]):
    #     frames[i], log_enery[i] = process_window(
    #         window=frames[i],
    #         dither=dither,
    #         remove_dc_offset=remove_dc_offset,
    #         preemphasis_coefficient=preemphasis_coefficient,
    #         window_function=feature_window_function(
    #             window_type=window_type,
    #             window_size=window_size,
    #             blackman_coeff=blackman_coeff
    #         ).astype(dtype),
    #         raw_energy=raw_energy
    #     )
        
    for i in range(frames.shape[0]):
        window_function = feature_window_function(
            window_type=window_type,
            window_size=window_size,
            blackman_coeff=blackman_coeff
        ).astype(dtype)
    # print(f"Shape of window_function at iteration {i}: {window_function.shape}") #(400,)
    # if(waveform[0] == 32768):
    #     print(window_function) #[0.08       0.08005703 0.08022812 0.08051322 0.08091226 0.08142514
    
        # print(f"frames[i].sjape {i}: {frames[i].shape}") #(400,)
        frames[i], log_enery[i] = process_window(
            window=frames[i],
            dither=dither,
            remove_dc_offset=remove_dc_offset,
            preemphasis_coefficient=preemphasis_coefficient,
            window_function=window_function,
            raw_energy=raw_energy
        )
    
    # print(frames.shape) #(23, 400),(11, 400),(4, 400)...(998, 400),(998, 400),(998, 400)...
    
    # if(waveform[0] == 32768):
    #     print(frames) #[-15689.318 -13077.196 -13026.279 ...  18126.3
    
    return frames, log_enery


# ---------- feature-window ----------


# ---------- feature-functions ----------

def compute_spectrum(frames, n):
    complex_spec = np.fft.rfft(frames, n)
    return np.absolute(complex_spec)


def compute_power_spectrum(frames, n):
    return np.square(compute_spectrum(frames, n))


# ---------- feature-functions ----------


# ---------- mel-computations ----------


def mel_scale(freq):
    return 1127.0 * np.log(1.0 + freq / 700.0)


def compute_mel_banks(num_bins, sample_frequency, low_freq, high_freq, n):
    """ Compute Mel banks.

    :param num_bins: Number of triangular mel-frequency bins
    :param sample_frequency: Waveform data sample frequency
    :param low_freq: Low cutoff frequency for mel bins
    :param high_freq: High cutoff frequency for mel bins (if <= 0, offset from Nyquist)
    :param n: Window size
    :return: Mel banks.
    """
    assert num_bins >= 3, 'Must have at least 3 mel bins'
    num_fft_bins = n // 2

    nyquist = 0.5 * sample_frequency
    if high_freq <= 0:
        high_freq = nyquist + high_freq
    assert 0 <= low_freq < high_freq <= nyquist

    fft_bin_width = sample_frequency / n

    mel_low_freq = mel_scale(low_freq)
    mel_high_freq = mel_scale(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    mel_banks = np.zeros([num_bins, num_fft_bins + 1])
    for i in range(num_bins):
        left_mel = mel_low_freq + mel_freq_delta * i
        center_mel = left_mel + mel_freq_delta
        right_mel = center_mel + mel_freq_delta
        for j in range(num_fft_bins):
            mel = mel_scale(fft_bin_width * j)
            if left_mel < mel < right_mel:
                if mel <= center_mel:
                    mel_banks[i, j] = (mel - left_mel) / (center_mel - left_mel)
                else:
                    mel_banks[i, j] = (right_mel - mel) / (right_mel - center_mel)
    return mel_banks


# ---------- mel-computations ----------


# ---------- compute-fbank-feats ----------

def compute_fbank_feats(
        waveform,
        blackman_coeff=0.42,
        dither=1.0,
        energy_floor=0.0,
        frame_length=25,
        frame_shift=10,
        high_freq=0,
        low_freq=20,
        num_mel_bins=23,
        preemphasis_coefficient=0.97,
        raw_energy=True,
        remove_dc_offset=True,
        round_to_power_of_two=True,
        sample_frequency=16000,
        snip_edges=True,
        use_energy=False,
        use_log_fbank=True,
        use_power=True,
        window_type='povey',
        dtype=np.float32):
    """ Compute (log) Mel filter bank energies

    :param waveform: Input waveform.
    :param blackman_coeff: Constant coefficient for generalized Blackman window. (float, default = 0.42)
    :param dither: Dithering constant (0.0 means no dither). If you turn this off, you should set the --energy-floor option, e.g. to 1.0 or 0.1 (float, default = 1)
    :param energy_floor: Floor on energy (absolute, not relative) in FBANK computation. Only makes a difference if --use-energy=true; only necessary if --dither=0.0.  Suggested values: 0.1 or 1.0 (float, default = 0)
    :param frame_length: Frame length in milliseconds (float, default = 25)
    :param frame_shift: Frame shift in milliseconds (float, default = 10)
    :param high_freq: High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (float, default = 0)
    :param low_freq: Low cutoff frequency for mel bins (float, default = 20)
    :param num_mel_bins: Number of triangular mel-frequency bins (int, default = 23)
    :param preemphasis_coefficient: Coefficient for use in signal preemphasis (float, default = 0.97)
    :param raw_energy: If true, compute energy before preemphasis and windowing (bool, default = true)
    :param remove_dc_offset: Subtract mean from waveform on each frame (bool, default = true)
    :param round_to_power_of_two: If true, round window size to power of two by zero-padding input to FFT. (bool, default = true)
    :param sample_frequency: Waveform data sample frequency (must match the waveform file, if specified there) (float, default = 16000)
    :param snip_edges: If true, end effects will be handled by outputting only frames that completely fit in the file, and the number of frames depends on the frame-length.  If false, the number of frames depends only on the frame-shift, and we reflect the data at the ends. (bool, default = true)
    :param use_energy: Add an extra energy output. (bool, default = false)
    :param use_log_fbank: If true, produce log-filterbank, else produce linear. (bool, default = true)
    :param use_power: If true, use power, else use magnitude. (bool, default = true)
    :param window_type: Type of window ("hamming"|"hanning"|"povey"|"rectangular"|"sine"|"blackmann") (string, default = "povey")
    :param dtype: Type of array (np.float32|np.float64) (dtype or string, default=np.float32)
    :return: (Log) Mel filter bank energies.
    """
    
    # print("waveform.shape {}".format(waveform.shape))
    
    window_size = int(frame_length * sample_frequency * 0.001)
    window_shift = int(frame_shift * sample_frequency * 0.001)
    
    # print(f"waveform.shape: {waveform.shape}") #(4001,),(2001,),(1001,),...(160000,),(160000,)...
    # print(f"blackman_coeff: {blackman_coeff}") #0.42
    # print(f"dither: {dither}") #0.0
    # print(f"window_size: {window_size}") #400
    # print(f"window_shift: {window_shift}") #160
    # print(f"preemphasis_coefficient: {preemphasis_coefficient}") #0.97
    # print(f"raw_energy: {raw_energy}") #True
    # print(f"remove_dc_offset: {remove_dc_offset}") #True
    # print(f"snip_edges: {snip_edges}") #True
    # print(f"window_type: {window_type}") #hamming
    # print(f"dtype: {dtype}") #<class 'numpy.float32'>
    
    frames, log_energy = extract_window(
        waveform=waveform,
        blackman_coeff=blackman_coeff,
        dither=dither,
        window_size=window_size,
        window_shift=window_shift,
        preemphasis_coefficient=preemphasis_coefficient,
        raw_energy=raw_energy,
        remove_dc_offset=remove_dc_offset,
        snip_edges=snip_edges,
        window_type=window_type,
        dtype=dtype
    )
    
    # print("frames.shape {}".format(frames.shape)) #(23, 400)，(11, 400)，(4, 400)，...(998, 400)，(998, 400)，(998, 400)...
    # dummyのときは(998, 400)
    
    # if(waveform[0] == 32768):
    #     print(frames) #[-15689.318 -13077.196 -13026.279 ...  18126.34   18166.38   18232.12 ] ...

    
    if round_to_power_of_two:
        n = 1
        while n < window_size:
            n *= 2
    else:
        n = window_size
    if use_power:
        # print("use_power") #呼ばれる
        spectrum = compute_power_spectrum(frames, n)
    else:
        spectrum = compute_spectrum(frames, n)
        
    
        
        
    # if(waveform[0] == 0.0 and waveform[1] == 0.0 and waveform[2] == 0.0): #waveformの1,2,3行目の条件
    #     # print("spectrum.shape {}".format(spectrum.shape)) #(998, 257),(998, 257),(998, 257)
    #     save_array_to_csv("spectrum012.csv", spectrum) #OKそう
    # if(waveform[0] == 24 and waveform[1] == 25 and waveform[2] == 22): #waveformの1,2,3行目の条件
    #     save_array_to_csv("spectrum345.csv", spectrum) #OKそう
        
        
        
    # print("num_mel_bins {}".format(num_mel_bins)) #80
    # print("sample_frequency {}".format(sample_frequency)) #16000
    # print("low_freq {}".format(low_freq)) #20
    # print("high_freq {}".format(high_freq)) #0
    # print("n {}".format(n)) #512
        
    mel_banks = compute_mel_banks(
        num_bins=num_mel_bins,
        sample_frequency=sample_frequency,
        low_freq=low_freq,
        high_freq=high_freq,
        n=n
    ).astype(dtype)
        
    # compute_mel_banksは良さそう
    # if(waveform[0] == 0.0 and waveform[1] == 0.0 and waveform[2] == 0.0): #waveformの1,2,3行目の条件
    #     # print("mel_banks.shape {}".format(mel_banks.shape)) #(80, 257), (80, 257), (80, 257)
    #     save_array_to_csv("mel_banks012.csv", mel_banks) #OK
    # if(waveform[0] == 24 and waveform[1] == 25 and waveform[2] == 22): #waveformの1,2,3行目の条件
    #     save_array_to_csv("mel_banks345.csv", mel_banks) #OK

    
    feat = np.dot(spectrum, mel_banks.T)
    # if(waveform[0] == 0.0 and waveform[1] == 0.0 and waveform[2] == 0.0): #waveformの1,2,3行目の条件
    #     save_array_to_csv("feat012.csv", feat) # //C++と全然違う
    
    if use_log_fbank:
        # print("use_log_fbank") #めちゃ呼ばれる
        feat = np.log(feat.clip(min=np.finfo(dtype).eps))
    if use_energy:
        # print("use_energy") #呼ばれない
        if energy_floor > 0.0:
            log_energy.clip(min=np.math.log(energy_floor))
        return feat, log_energy
    
    
    # print(feat.shape) #(998, 80)
    # if(waveform[0] == 0.0 and waveform[1] == 0.0 and waveform[2] == 0.0): #waveformの1,2,3行目の条件
    #     save_array_to_csv("feat012.csv", feat) # //OK

    
    
    


    
    # waveformの0,1,2行目は同じ，3,4,5行目は同じ...という構造
    # if(waveform[0] == 0.0 and waveform[1] == 0.0 and waveform[2] == 0.0): #waveformの一行目の条件
    #     print("feat.shape {}".format(feat.shape)) #feat.shape (998, 80)
    # elif(waveform[0] == -391.0):
    #     print("feat.shape {}".format(feat.shape)) #feat.shape (998, 80)
        
    # だからfeatも，feat0, feat1，feat2は同じ，feat3, feat4，feat5は同じ
    
    return feat

# ---------- compute-fbank-feats ----------
