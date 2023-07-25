import sys
import time
from logging import getLogger

import numpy as np
import scipy.signal as signal
import torch
import torch.nn.functional as F
import ffmpeg
import librosa
import soundfile as sf

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
    '--sid', type=int, default=0,
    help='Select Speaker/Singer ID',
)
parser.add_argument(
    '--index_rate', type=float, default=0.75,
    help='Search feature ratio. (controls accent strength, too high has artifacting)',
)
parser.add_argument(
    '--resample_sr', type=int, default=0,
    help='Resample the output audio. Set to 0 for no resampling.',
)
parser.add_argument(
    '--rms_mix_rate', type=float, default=0.25,
    help='Adjust the volume envelope scaling.',
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


def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
            torch.pow(rms1, torch.tensor(1 - rate))
            * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()

    return data2


# ======================
# Main functions
# ======================

def vc(
        hubert,
        net_g,
        sid,
        audio0,
        vc_param,
        index,
        big_npy,
        index_rate):
    feats = audio0.reshape(1, -1).astype(np.float32)
    padding_mask = np.zeros(feats.shape, dtype=bool)

    # feedforward
    if not args.onnx:
        output = hubert.predict([feats, padding_mask])
    else:
        output = hubert.run(None, {'source': feats, 'padding_mask': padding_mask})
    feats = output[0]

    if isinstance(index, type(None)) is False \
            and isinstance(big_npy, type(None)) is False \
            and index_rate != 0:
        x = feats[0]

        score, ix = index.search(x, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        x = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

        feats = (
                np.expand_dims(x, axis=0) * index_rate
                + (1 - index_rate) * feats
        )

    feats = torch.from_numpy(feats)
    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    feats = feats.numpy()

    p_len = audio0.shape[0] // vc_param.window
    if feats.shape[1] < p_len:
        p_len = feats.shape[1]
    p_len = np.array([p_len], dtype=int)

    # feedforward
    rnd = np.random.randn(1, 192, p_len[0]).astype(np.float32) * 0.66666  # 噪声（加入随机因子）
    if not args.onnx:
        output = net_g.predict([feats, p_len, sid, rnd])
    else:
        output = net_g.run(None, {
            'phone': feats, 'phone_lengths': p_len, 'ds': sid, 'rnd': rnd
        })
    audio1 = output[0][0, 0]

    return audio1


bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def predict(audio, models, tgt_sr):
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1:
        audio /= audio_max

    sid = args.sid
    index_rate = args.index_rate
    resample_sr = args.resample_sr
    rms_mix_rate = args.rms_mix_rate

    vc_param = VCParam(tgt_sr)

    index = big_npy = None
    audio = signal.filtfilt(bh, ah, audio)
    audio_pad = np.pad(audio, (vc_param.window // 2, vc_param.window // 2), mode="reflect")

    opt_ts = []
    if audio_pad.shape[0] > vc_param.t_max:
        audio_sum = np.zeros_like(audio)
        for i in range(vc_param.window):
            audio_sum += audio_pad[i: i - vc_param.window]
        for t in range(vc_param.t_center, audio.shape[0], vc_param.t_center):
            opt_ts.append(
                t - vc_param.t_query
                + np.where(
                    np.abs(audio_sum[t - vc_param.t_query: t + vc_param.t_query])
                    == np.abs(audio_sum[t - vc_param.t_query: t + vc_param.t_query]).min()
                )[0][0]
            )

    s = 0
    audio_opt = []
    t = None
    audio_pad = np.pad(audio, (vc_param.t_pad, vc_param.t_pad), mode="reflect")

    sid = np.array([sid], dtype=int)
    for t in opt_ts:
        t = t // vc_param.window * vc_param.window
        audio1 = vc(
            models["hubert"],
            models["net_g"],
            sid,
            audio_pad[s: t + vc_param.t_pad2 + vc_param.window],
            vc_param,
            index,
            big_npy,
            index_rate,
        )
        audio_opt.append(audio1[vc_param.t_pad_tgt: -vc_param.t_pad_tgt])
        s = t
    audio1 = vc(
        models["hubert"],
        models["net_g"],
        sid,
        audio_pad[t:],
        vc_param,
        index,
        big_npy,
        index_rate,
    )
    audio_opt.append(audio1[vc_param.t_pad_tgt: -vc_param.t_pad_tgt])
    audio_opt = np.concatenate(audio_opt)
    audio_opt = audio_opt.astype(np.float32)

    if rms_mix_rate < 1:
        audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
    if 16000 <= resample_sr != tgt_sr:
        audio_opt = librosa.resample(
            audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
        )
        tgt_sr = resample_sr

    audio_max = np.abs(audio_opt).max() / 0.99
    max_int16 = 32768
    if audio_max > 1:
        max_int16 /= audio_max
    audio_opt = (audio_opt * max_int16).astype(np.int16)

    return audio_opt, tgt_sr


def recognize_from_audio(models):
    # Depend on voice model
    tgt_sr = 40000

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
            output, sr = predict(audio, models, tgt_sr)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\ttotal processing time {estimation_time} ms')
        else:
            output, sr = predict(audio, models, tgt_sr)

        # save result
        savepath = get_savepath(args.savepath, audio_path, ext='.wav')
        logger.info(f'saved at : {savepath}')
        sf.write(savepath, output, sr)

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
