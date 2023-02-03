import sys
import time
from logging import getLogger

import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

from beam_search import BatchBeamSearch

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'reazonspeech-espnet-v1.onnx'
MODEL_PATH = 'reazonspeech-espnet-v1.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/reason_speech/'

WAV_PATH = 'speech-001.wav'
SAVE_TEXT_PATH = 'output.txt'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ReazonSpeech', WAV_PATH, SAVE_TEXT_PATH, input_ftype='audio'
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

# def draw_bbox(img, bboxes):
#     return img


# ======================
# Main functions
# ======================

def decode(beam_search, enc):
    maxlenratio = minlenratio = 0.0
    nbest_hyps = beam_search.forward(
        x=enc, maxlenratio=maxlenratio, minlenratio=minlenratio
    )

    return None


def predict(mod, wav):
    speech = np.load("speech-001.npy")
    lengths = np.load("lengths-001.npy")
    # speech = np.load("speech-002.npy")
    # lengths = np.load("lengths-002.npy")
    # print(speech.shape)
    print(lengths.shape)

    net = mod['net']

    # feedforward
    if not args.onnx:
        output = net.predict([speech, lengths])
    else:
        output = net.run(None, {'speech': speech, 'lengths': lengths})
    enc, _ = output

    print(enc)
    print(enc.shape)

    beam_search = mod['beam_search']

    decode(beam_search, enc[0])

    return enc


def recognize_from_audio(mod):
    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        # wav = load_audio(audio_path)
        wav = None

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(mod, wav)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(mod, wav)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    with open('config.yaml') as file:
        import yaml
        config = yaml.safe_load(file)

    beam_size = 20
    weights = {'decoder': 0.5, 'ctc': 0.5, 'lm': 1.0, 'ngram': 0.9, 'length_bonus': 0.0}
    scorers = None
    sos = eos = 2601
    token_list = config['token_list']

    beam_search = BatchBeamSearch(
        beam_size=beam_size,
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        vocab_size=len(token_list),
        token_list=token_list,
        pre_beam_score_key='full'
    )

    mod = {
        'net': net,
        'beam_search': beam_search
    }

    recognize_from_audio(mod)


if __name__ == '__main__':
    main()
