import sys
import time
from logging import getLogger

import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'rnn_model.onnx'
MODEL_PATH = 'rnn_model.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/rnnoise/'

AUDIO_PATH = 'babble_15dB.wav'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'rnnoise', AUDIO_PATH, None
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

NB_BANDS = 22
NB_DELTA_CEPS = 6
NB_FEATURES = NB_BANDS + 3 * NB_DELTA_CEPS + 2

FRAME_SIZE_SHIFT = 2
FRAME_SIZE = 120 << FRAME_SIZE_SHIFT


class CommonState:
    init = False
    kfft = None
    half_window = np.array(FRAME_SIZE)
    dct_table = np.array(NB_BANDS * NB_BANDS)


import math


def check_init():
    if CommonState.init:
        return

    # common.kfft = opus_fft_alloc_twiddles(2 * FRAME_SIZE, NULL, NULL, NULL, 0);

    for i in range(FRAME_SIZE):
        CommonState.half_window[i] = math.sin(
            .5 * math.M_PI * math.sin(.5 * math.M_PI * (i + .5) / FRAME_SIZE)
            * math.sin(.5 * math.M_PI * (i + .5) / FRAME_SIZE)
        )

    for i in range(NB_BANDS):
        for j in range(NB_BANDS):
            CommonState.dct_table[i * NB_BANDS + j] = math.cos((i + .5) * j * math.M_PI / NB_BANDS)
            if j == 0:
                CommonState.dct_table[i * NB_BANDS + j] *= math.sqrt(.5)

    CommonState.init = True


def dct(in_data):
    check_init()

    out = []
    for i in range(NB_BANDS):
        _sum = 0
        for j in range(NB_BANDS):
            _sum += in_data[j] * CommonState.dct_table[j * NB_BANDS + i]
        out.append(_sum * math.sqrt(2. / 22))

    return out


def compute_frame_features():
    Ly = np.zeros(NB_BANDS)

    features = np.zeros(NB_FEATURES)
    pitch_index = 375

    # pitch_search(
    #     pitch_buf + (PITCH_MAX_PERIOD >> 1), pitch_buf, PITCH_FRAME_SIZE,
    #     PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD, & pitch_index)
    # pitch_index = PITCH_MAX_PERIOD - pitch_index

    features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3
    features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9
    features[NB_BANDS + 3 * NB_DELTA_CEPS] = .01 * (pitch_index - 300)


# ======================
# Main functions
# ======================


def rnnoise_process_frame(net, x):
    pass


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    import wave
    import struct
    wf = wave.open("babble_15dB.wav", "rb")

    FRAME_SIZE = 480
    while True:
        buf = wf.readframes(FRAME_SIZE)
        if not buf:
            break
        x = [struct.unpack("<h", buf[i * 2:i * 2 + 2])[0] for i in range(len(buf) // 2)]

        rnnoise_process_frame(net, x)


if __name__ == '__main__':
    main()
