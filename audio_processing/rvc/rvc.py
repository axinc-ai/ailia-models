import os
import sys
import time
import queue
from logging import getLogger

import numpy as np
import scipy.signal as signal
import ffmpeg

import ailia

# import original modules
sys.path.append('../../util')
from math_utils import softmax
from microphone_utils import start_microphone_input  # noqa
from model_utils import check_and_download_models  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_HUBERT_PATH = "hubert_base.onnx"
MODEL_HUBERT_PATH = "hubert_base.onnx.prototxt"
WEIGHT_AISO_HOWATTO_PATH = "AISO-HOWATTO.onnx"
MODEL_AISO_HOWATTO_PATH = "AISO-HOWATTO.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/rvc/'

WAV_PATH = 'demo.wav'
SAVE_TEXT_PATH = 'output.txt'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Retrieval-based-Voice-Conversion', WAV_PATH, SAVE_TEXT_PATH, input_ftype='audio'
)
parser.add_argument(
    '-V', action='store_true',
    help='use microphone input',
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


class VCParam(object):
    def __init__(self, tgt_sr):
        self.x_pad, self.x_query, self.x_center, self.x_max = (
            3, 10, 60, 65
        )
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值


# ======================
# Secondaty Functions
# ======================

def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = ffmpeg.input(file, threads=0) \
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr) \
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


# ======================
# Main functions
# ======================

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def predict(audio, models):
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1:
        audio /= audio_max

    tgt_sr = 40000
    vcp = VCParam(tgt_sr)

    index = big_npy = None
    audio = signal.filtfilt(bh, ah, audio)
    audio_pad = np.pad(audio, (vcp.window // 2, vcp.window // 2), mode="reflect")

    return None


def recognize_from_audio(models):
    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        audio = load_audio(audio_path, 16000)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            start = int(round(time.time() * 1000))
            output = predict(audio, models)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\ttotal processing time {estimation_time} ms')
        else:
            output = predict(audio, models)

    logger.info('Script finished successfully.')


def main():
    env_id = args.env_id

    # initialize
    if not args.onnx:
        hubert = ailia.Net(MODEL_HUBERT_PATH, WEIGHT_HUBERT_PATH, env_id=env_id)
        net_g = ailia.Net(MODEL_AISO_HOWATTO_PATH, WEIGHT_AISO_HOWATTO_PATH, env_id=env_id)
    else:
        import onnxruntime
        providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        hubert = onnxruntime.InferenceSession(WEIGHT_HUBERT_PATH, providers=providers)
        net_g = onnxruntime.InferenceSession(WEIGHT_AISO_HOWATTO_PATH, providers=providers)

    models = {
        "hubert": hubert,
        "net_g": net_g,
    }

    recognize_from_audio(models)


if __name__ == '__main__':
    main()
