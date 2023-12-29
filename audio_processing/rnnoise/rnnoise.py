import sys
import time
import math
from logging import getLogger

import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

from kiss_fft import Complex, opus_fft_alloc_twiddles, opus_fft

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'rnn_model.onnx'
MODEL_PATH = 'rnn_model.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/rnnoise/'

AUDIO_PATH = 'babble_15dB.wav'

PITCH_MAX_PERIOD = 768
PITCH_FRAME_SIZE = 960
PITCH_BUF_SIZE = PITCH_MAX_PERIOD + PITCH_FRAME_SIZE

NB_BANDS = 22
CEPS_MEM = 8
NB_DELTA_CEPS = 6
NB_FEATURES = NB_BANDS + 3 * NB_DELTA_CEPS + 2

FRAME_SIZE_SHIFT = 2
FRAME_SIZE = 120 << FRAME_SIZE_SHIFT
WINDOW_SIZE = 2 * FRAME_SIZE
FREQ_SIZE = FRAME_SIZE + 1

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


class CommonState:
    init = False
    kfft = None
    half_window = np.zeros(FRAME_SIZE)
    dct_table = np.zeros(NB_BANDS * NB_BANDS)


common = CommonState()


class DenoiseState:
    analysis_mem = np.zeros(FRAME_SIZE)
    cepstral_mem = np.zeros((CEPS_MEM, NB_BANDS))
    memid = 0
    synthesis_mem = np.zeros(FRAME_SIZE)
    pitch_buf = np.zeros(PITCH_BUF_SIZE)
    pitch_enh_buf = np.zeros(PITCH_BUF_SIZE)
    last_gain = 0.0
    last_period = 0
    mem_hp_x = np.zeros(2)
    lastg = np.zeros(NB_BANDS)


def check_init():
    if common.init:
        return

    common.kfft = opus_fft_alloc_twiddles(2 * FRAME_SIZE)

    for i in range(FRAME_SIZE):
        common.half_window[i] = math.sin(
            .5 * math.pi * math.sin(.5 * math.pi * (i + .5) / FRAME_SIZE)
            * math.sin(.5 * math.pi * (i + .5) / FRAME_SIZE)
        )

    for i in range(NB_BANDS):
        for j in range(NB_BANDS):
            common.dct_table[i * NB_BANDS + j] = math.cos((i + .5) * j * math.pi / NB_BANDS)
            if j == 0:
                common.dct_table[i * NB_BANDS + j] *= math.sqrt(.5)

    common.init = True


def dct(in_data):
    check_init()

    out = []
    for i in range(NB_BANDS):
        _sum = 0
        for j in range(NB_BANDS):
            _sum += in_data[j] * common.dct_table[j * NB_BANDS + i]
        out.append(_sum * math.sqrt(2. / 22))

    return out


def forward_transform(out, in_data):
    check_init()

    x = [Complex() for _ in range(WINDOW_SIZE)]
    y = [Complex() for _ in range(WINDOW_SIZE)]

    for i in range(WINDOW_SIZE):
        x[i].r = in_data[i]
        x[i].i = 0

    opus_fft(common.kfft, x, y)
    for i in range(FREQ_SIZE):
        out[i] = y[i]


def apply_window(x):
    check_init()

    for i in range(FRAME_SIZE):
        x[i] *= common.half_window[i]
        x[WINDOW_SIZE - 1 - i] *= common.half_window[i]


def frame_analysis(st, X, in_data):
    x = np.zeros(WINDOW_SIZE)
    x[:FRAME_SIZE] = st.analysis_mem
    x[FRAME_SIZE:] = in_data
    st.analysis_mem[...] = in_data

    apply_window(x)
    forward_transform(X, x)

    # compute_band_energy(Ex, X)


def compute_frame_features(st, X, P, x):
    E = 0
    Ly = np.zeros(NB_BANDS)

    frame_analysis(st, X, x)

    features = np.zeros(NB_FEATURES)
    pitch_index = 375

    # pitch_search(
    #     pitch_buf + (PITCH_MAX_PERIOD >> 1), pitch_buf, PITCH_FRAME_SIZE,
    #     PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD, & pitch_index)
    # pitch_index = PITCH_MAX_PERIOD - pitch_index

    features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3
    features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9
    features[NB_BANDS + 3 * NB_DELTA_CEPS] = .01 * (pitch_index - 300)

    logMax = -2
    follow = -2
    for i in range(NB_BANDS):
        Ly[i] = math.log10(1e-2 + Ex[i])
        Ly[i] = MAX16(logMax - 7, MAX16(follow - 1.5, Ly[i]))
        logMax = MAX16(logMax, Ly[i])
        follow = MAX16(follow - 1.5, Ly[i])
        E += Ex[i]

    if E < 0.04:
        # If there's no audio, avoid messing up the state.
        RNN_CLEAR(features, NB_FEATURES)
        return 1

    dct(features, Ly)


def biquad(y, mem, x, b, a, N):
    for i in range(N):
        xi = x[i]
        yi = x[i] + mem[0]
        mem[0] = mem[1] + (b[0] * xi - a[0] * yi)
        mem[1] = b[1] * xi - a[1] * yi
        y[i] = yi


# ======================
# Main functions
# ======================


def rnnoise_process_frame(net, st, in_data):
    X = [Complex() for _ in range(FREQ_SIZE)]
    P = [Complex() for _ in range(WINDOW_SIZE)]

    x = np.zeros(FRAME_SIZE)

    a_hp = (-1.99599, 0.99600)
    b_hp = (-2., 1.)
    biquad(x, st.mem_hp_x, in_data, b_hp, a_hp, FRAME_SIZE)
    compute_frame_features(st, X, P, x)


def recognize_from_audio(net):
    import wave
    import struct
    wf = wave.open("babble_15dB.wav", "rb")

    st = DenoiseState()
    while True:
        buf = wf.readframes(FRAME_SIZE)
        if not buf:
            break
        x = [struct.unpack("<h", buf[i * 2:i * 2 + 2])[0] for i in range(len(buf) // 2)]

        rnnoise_process_frame(net, st, x)


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

    recognize_from_audio(net)


if __name__ == '__main__':
    main()
