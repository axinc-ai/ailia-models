import os
import sys
import time

import numpy as np
import librosa
import soundfile as sf

import ailia
from audio_utils import Audio

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'model.onnx'
MODEL_PATH = 'model.onnx.prototxt'
WEIGHT_EMB_PATH = 'embedder.onnx'
MODEL_EMB_PATH = 'embedder.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/voicefilter/'

WAVE_PATH = "mixed.wav"
SAVE_PATH = 'output.wav'

# Audio
SAMPLING_RATE = 16000

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'VoiceFilter', WAVE_PATH, SAVE_PATH, input_ftype='audio'
)
parser.add_argument(
    '-r', '--reference_file',
    default="ref-voice.wav", type=str,
    help='path of reference wav file'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def read_wave(path):
    # prepare input data
    wav, source_sr = librosa.load(path, sr=None)
    # Resample the wav if needed
    if source_sr is not None and source_sr != SAMPLING_RATE:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=SAMPLING_RATE)

    return wav


# ======================
# Main functions
# ======================

def audio_recognition(net, embedder):
    reference_file = args.reference_file
    if not reference_file or not os.path.exists(reference_file):
        logger.error('reference_file:%s is NG.' % reference_file)
        sys.exit(-1)

    audio = Audio()

    # prepare reference wav
    dvec_wav = read_wave(reference_file)
    dvec_mel = audio.get_mel(dvec_wav)
    output = embedder.predict([dvec_mel])
    dvec = output[0]
    dvec = np.expand_dims(dvec, axis=0)

    for soundf_path in args.input:
        logger.info(soundf_path)

        # prepare mix wav
        mixed_wav = read_wave(soundf_path)
        mag, phase = audio.wav2spec(mixed_wav)
        mag = np.expand_dims(mag, axis=0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict([mag, dvec])
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = net.predict([mag, dvec])

        mask = output[0]

        est_mag = mag * mask
        est_wav = audio.spec2wav(est_mag[0], phase)

        savepath = get_savepath(args.savepath, soundf_path, ext='.wav')
        logger.info(f'saved at : {savepath}')
        sf.write(savepath, est_wav, SAMPLING_RATE, 'PCM_24')

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('Checking voicefilter model...')
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    logger.info('Checking embedder model...')
    check_and_download_models(WEIGHT_EMB_PATH, MODEL_EMB_PATH, REMOTE_PATH)

    env_id = args.env_id

    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    embedder = ailia.Net(MODEL_EMB_PATH, WEIGHT_EMB_PATH, env_id=env_id)

    audio_recognition(net, embedder)


if __name__ == '__main__':
    main()
