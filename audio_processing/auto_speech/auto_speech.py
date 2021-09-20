import sys
import time
import random

import numpy as np
import soundfile as sf
import librosa

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_IDENT_PATH = './proposed_iden.onnx'
MODEL_IDENT_PATH = './proposed_iden.onnx.prototxt'
WEIGHT_CLASSIFIER_PATH = './proposed_iden_classifier.onnx'
MODEL_CLASSIFIER_PATH = './proposed_iden_classifier.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/auto_speech/'

WAVE_PATH = "wav/id10283/oGZsanLiXsY/00004.wav"

# Audio
SAMPLING_RATE = 16000

# Mel-filterbank
WINDOW_LENGTH = 25  # In milliseconds
WINDOW_STEP = 10  # In milliseconds
N_FFT = 512

# Audio volume normalization
AUDIO_NORM_TARGET_dBFS = -30

INT16_MAX = (2 ** 15) - 1

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'AutoSpeech', WAVE_PATH, None, input_ftype='audio'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def voxceleb1_ids():
    with open("VoxCeleb1_ids.txt") as f:
        ids = [x.strip() for x in f]

    return ids


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * INT16_MAX) ** 2))
    wave_dBFS = 20 * np.log10(rms / INT16_MAX)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def wav_to_spectrogram(wav):
    frames = np.abs(librosa.core.stft(
        wav,
        n_fft=N_FFT,
        hop_length=int(SAMPLING_RATE * WINDOW_STEP / 1000),
        win_length=int(SAMPLING_RATE * WINDOW_LENGTH / 1000),
    ))
    return frames.astype(np.float32).T


def generate_sequence(feature, partial_n_frames, shift=None):
    while feature.shape[0] <= partial_n_frames:
        feature = np.repeat(feature, 2, axis=0)
    if shift is None:
        shift = partial_n_frames // 2
    test_sequence = []
    start = 0
    while start + partial_n_frames <= feature.shape[0]:
        test_sequence.append(feature[start: start + partial_n_frames])
        start += shift
    test_sequence = np.stack(test_sequence, axis=0)
    return test_sequence


# ======================
# Main functions
# ======================

def preprocess(wav):
    wav = normalize_volume(wav, AUDIO_NORM_TARGET_dBFS, increase_only=True)
    feature = wav_to_spectrogram(wav)

    sequence = generate_sequence(feature, partial_n_frames=300)

    mean = np.load('mean.npy')
    std = np.load('std.npy')
    sequence = (sequence - mean) / std
    # if random.random() < 0.5:
    #     sequence = np.flip(sequence, axis=0).copy()

    return sequence


def predict(wav, net, net_classifier):
    # initial preprocesses
    sequence = preprocess(wav)

    # feedforward
    output = net.predict([sequence])
    output = output[0]

    output = np.mean(output, axis=0, keepdims=True)
    output = net_classifier.predict([output])
    output = output[0]

    idx = np.argsort(output[0])[::-1]

    return idx


def recognize_from_audio(net, net_classifier):
    ids = voxceleb1_ids()

    for input_path in args.input:
        logger.info(f'input: {input_path}')

        # prepare input data
        # wav = sf.read(input_path)
        wav, source_sr = librosa.load(input_path, sr=None)
        # Resample the wav if needed
        if source_sr is not None and source_sr != SAMPLING_RATE:
            wav = librosa.resample(wav, source_sr, SAMPLING_RATE)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                idx = predict(wav, net, net_classifier)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            idx = predict(wav, net, net_classifier)

        logger.info(' Top5: %s' % ', '.join([ids[i] for i in idx[:5]]))

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_IDENT_PATH, MODEL_IDENT_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_CLASSIFIER_PATH, MODEL_CLASSIFIER_PATH, REMOTE_PATH)

    # load model
    env_id = ailia.get_gpu_environment_id()
    logger.info(f'env_id: {env_id}')

    # initialize
    net = ailia.Net(MODEL_IDENT_PATH, WEIGHT_IDENT_PATH, env_id=env_id)
    net_classifier = ailia.Net(MODEL_CLASSIFIER_PATH, WEIGHT_CLASSIFIER_PATH, env_id=env_id)

    recognize_from_audio(net, net_classifier)


if __name__ == '__main__':
    main()
