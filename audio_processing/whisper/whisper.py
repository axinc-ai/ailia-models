import os
import sys
import time

import numpy as np
import tqdm

import ailia
import ailia.audio

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
# logger
from logging import getLogger  # noqa

from audio_utils import SAMPLE_RATE, HOP_LENGTH, N_FRAMES
from audio_utils import load_audio, log_mel_spectrogram, pad_or_trim

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_COND_STAGE_PATH = 'xxx.onnx'
MODEL_COND_STAGE_PATH = 'xxx.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/whisper/'

WAV_PATH = 'demo.png'
SAVE_TEXT_PATH = 'output.txt'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Whisper', WAV_PATH, SAVE_TEXT_PATH, input_ftype='audio'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


# ======================
# Main functions
# ======================

def predict(wav, enc_net, dec_net):
    mel = log_mel_spectrogram(wav)
    mel = np.expand_dims(mel, axis=0)

    # language = "Japanese"
    # task = "transcribe"
    # tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    seek = 0

    num_frames = mel.shape[-1]
    previous_seek_value = seek
    with tqdm.tqdm(total=num_frames, unit='frames') as pbar:
        while seek < num_frames:
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, :, seek:], N_FRAMES)
            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

            # update progress bar
            pbar.update(min(num_frames, seek) - previous_seek_value)
            previous_seek_value = seek

            break

    # if not args.onnx:
    #     output = enc_net.predict([x])
    # else:
    #     output = enc_net.run(None, {'masked_image': x})
    # c = output[0]

    return


def recognize_from_audio(enc_net, dec_net):
    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        wav = load_audio(audio_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(wav, enc_net, dec_net)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(wav, enc_net, dec_net)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.txt')
        logger.info(f'saved at : {savepath}')

    logger.info('Script finished successfully.')


def main():
    WEIGHT_ENC_SMALL_PATH = "encoder_small.onnx"
    MODEL_ENC_SMALL_PATH = "encoder_small.onnx.prototxt"
    WEIGHT_DEC_SMALL_PATH = "decoder_small.onnx"
    MODEL_DEC_SMALL_PATH = "decoder_small.onnx.prototxt"
    check_and_download_models(WEIGHT_ENC_SMALL_PATH, MODEL_ENC_SMALL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_SMALL_PATH, MODEL_DEC_SMALL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        enc_net = ailia.Net(MODEL_ENC_SMALL_PATH, WEIGHT_ENC_SMALL_PATH, env_id=env_id)
        dec_net = ailia.Net(MODEL_DEC_SMALL_PATH, WEIGHT_DEC_SMALL_PATH, env_id=env_id)
    else:
        import onnxruntime
        enc_net = onnxruntime.InferenceSession(WEIGHT_ENC_SMALL_PATH)
        dec_net = onnxruntime.InferenceSession(WEIGHT_DEC_SMALL_PATH)

    recognize_from_audio(enc_net, dec_net)


if __name__ == '__main__':
    main()
