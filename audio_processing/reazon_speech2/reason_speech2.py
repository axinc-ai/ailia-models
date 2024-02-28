import sys
import time
from logging import getLogger

import yaml
import numpy as np
import librosa

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_ENC_PATH = 'reazonspeech-nemo-v2_encoder.onnx'
MODEL_ENC_PATH = 'reazonspeech-nemo-v2_encoder.onnx.prototxt'
WEIGHT_DEC_PATH = 'reazonspeech-nemo-v2_decoder.onnx'
MODEL_DEC_PATH = 'reazonspeech-nemo-v2_decoder.onnx.prototxt'
WEIGHT_JNT_PATH = 'reazonspeech-nemo-v2_joint.onnx'
MODEL_JNT_PATH = 'reazonspeech-nemo-v2_joint.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/reason_speech2/'

WAV_PATH = 'speech-001.wav'
SAVE_TEXT_PATH = 'output.txt'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ReazonSpeech2', WAV_PATH, SAVE_TEXT_PATH, input_ftype='audio'
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

def decode(models, enc):
    nbest = args.nbest
    maxlenratio = args.maxlenratio
    minlenratio = args.minlenratio

    beam_search = mod['beam_search']
    token_list = mod['token_list']
    tokenizer = mod['tokenizer']

    nbest_hyps = beam_search.forward(
        x=enc, maxlenratio=maxlenratio, minlenratio=minlenratio
    )

    nbest_hyps = nbest_hyps[: nbest]

    results = []
    for hyp in nbest_hyps:
        # remove sos/eos and get results
        token_int = hyp.yseq[1:-1].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x != 0, token_int))

        # Change integer-ids to tokens
        token = [token_list[i] for i in token_int]

        text = tokenizer.tokens2text(token)
        results.append((text, token, token_int, hyp))

    return results


def predict(models, audio):
    input_signal_length = len(audio)
    input_signal_length = np.array([input_signal_length])
    audio = np.expand_dims(audio, axis=0)
    net = models['encoder']

    # feedforward
    if not args.onnx:
        output = net.predict([audio, input_signal_length])
    else:
        output = net.run(None, {'input_signal': audio, 'input_signal_length': input_signal_length})
    encoded, encoded_length = output

    results = decode(models, enc[0])

    return results


def recognize_from_audio(models):
    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        audio, rate = librosa.load(audio_path, sr=16000)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(models, audio)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(models, audio)

        for res in output:
            logger.info(f"{res[0]}")

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_JNT_PATH, MODEL_JNT_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        encoder = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id)
        decoder = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id)
        joint = ailia.Net(MODEL_JNT_PATH, WEIGHT_JNT_PATH, env_id=env_id)
    else:
        import onnxruntime
        encoder = onnxruntime.InferenceSession(WEIGHT_ENC_PATH)
        decoder = onnxruntime.InferenceSession(WEIGHT_DEC_PATH)
        joint = onnxruntime.InferenceSession(WEIGHT_JNT_PATH)

    models = {
        'encoder': encoder,
        'decoder': decoder,
        'joint': joint,
    }

    recognize_from_audio(models)


if __name__ == '__main__':
    main()
