import sys
import time
from logging import getLogger

import numpy as np
import scipy.signal as signal
from PIL import Image
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from microphone_utils import start_microphone_input  # noqa
from model_utils import check_and_download_models  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

# crepe util
import mod_crepe
from mod_crepe import WEIGHT_CREPE_PATH, MODEL_CREPE_PATH, WEIGHT_CREPE_TINY_PATH, MODEL_CREPE_TINY_PATH

flg_ffmpeg = False

if flg_ffmpeg:
    import ffmpeg

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/rvc/'

SAMPLE_RATE = 16000

WAV_PATH = 'booth.wav'
FIG_PATH = "output.png"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Crepe', WAV_PATH, FIG_PATH, input_ftype='audio'
)
parser.add_argument(
    '--f0_method', default="crepe_tiny", choices=("pm", "harvest", "crepe", "crepe_tiny"),
    help='Select the pitch extraction algorithm',
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

def load_audio(file: str, sr: int = SAMPLE_RATE):
    if flg_ffmpeg:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = ffmpeg.input(file, threads=0) \
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr) \
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)

        audio = np.frombuffer(out, np.float32).flatten()
    else:
        # prepare input data
        audio, source_sr = librosa.load(file, sr=None)
        # Resample the wav if needed
        if source_sr is not None and source_sr != sr:
            audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

    return audio


# ======================
# Main functions
# ======================

def get_f0(
        f0_method,
        window,
        x,
        p_len,
):
    sr = SAMPLE_RATE
    time_step = window / sr * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    if f0_method == "pm":
        import parselmouth

        f0 = (
            parselmouth.Sound(x, vc_param.sr).to_pitch_ac(
                time_step=time_step / 1000,
                voicing_threshold=0.6,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            ).selected_array["frequency"]
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(
                f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
            )
    elif f0_method == "harvest":
        import pyworld

        audio = x.astype(np.double)
        fs = sr
        frame_period = 10
        f0, t = pyworld.harvest(
            audio,
            fs=fs,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=frame_period,
        )
        f0 = pyworld.stonemask(audio, f0, t, fs)

        filter_radius = 3
        if filter_radius > 2:
            f0 = signal.medfilt(f0, 3)
    elif f0_method == "crepe" or f0_method == "crepe_tiny":
        import mod_crepe

        # Pick a batch size that doesn't cause memory errors on your gpu
        batch_size = 512
        audio = np.copy(x)[None]
        f0, pd = mod_crepe.predict(
            audio,
            sr,
            window,
            f0_min,
            f0_max,
            batch_size=batch_size,
            return_periodicity=True,
        )
        pd = mod_crepe.median(pd, 3)
        f0 = mod_crepe.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0]
    else:
        raise ValueError("f0_method: %s" % f0_method)

    return f0


def predict(audio, models):
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1:
        audio /= audio_max

    window = 160       
    p_len = audio.shape[0] // window

    pitch = get_f0(
        args.f0_method,
        window,
        audio,
        p_len,
    )

    return pitch



def recognize_from_audio(models):
    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        audio = load_audio(audio_path, SAMPLE_RATE)

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
        
        # plot
        x = np.linspace(0, 1, output.shape[0])
        y = output
        plt.plot(x, y, label="f0 (hz)")
        plt.legend()
        
        # save result
        savepath = get_savepath(args.savepath, audio_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        plt.savefig(savepath)

    logger.info('Script finished successfully.')


def main():
    if args.f0_method == "crepe_tiny":
        check_and_download_models(WEIGHT_CREPE_TINY_PATH, MODEL_CREPE_TINY_PATH, REMOTE_PATH)
    else:
        check_and_download_models(WEIGHT_CREPE_PATH, MODEL_CREPE_PATH, REMOTE_PATH)

    env_id = args.env_id

    f0_model = mod_crepe.load_model(env_id, args.onnx, args.f0_method == "crepe_tiny")
    if args.profile:
        f0_model.set_profile_mode(True)
    else:
        f0_model = None

    recognize_from_audio(f0_model)

    if args.profile and not args.onnx:
        print("--- profile f0_model")
        print(f0_model.get_summary())
        print("")


if __name__ == '__main__':
    main()
