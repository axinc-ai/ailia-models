from collections import namedtuple
from fractions import Fraction
import sys
import time
from logging import getLogger
from typing import Optional

import numpy as np
import scipy.signal as signal
from PIL import Image
import librosa
import soundfile as sf

import ailia

# import original modules
sys.path.append("../../util")
from model_utils import check_and_download_models  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

flg_ffmpeg = False

if flg_ffmpeg:
    import ffmpeg

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/demucs/"

SAMPLE_RATE = 16000

WAV_PATH = "test.mp3"
SAVE_WAV_PATH = "output.wav"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "Demucs Music Source Separation", WAV_PATH, SAVE_WAV_PATH, input_ftype="audio"
)
parser.add_argument(
    "-m", "--model_file", default="htdemucs_ft_drums.onnx", help="specify .onnx file"
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


TensorChunk = namedtuple("TensorChunk", ["tensor", "offset", "length"])


def load_audio(file: str, sr: int = SAMPLE_RATE):
    if flg_ffmpeg:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )

        audio = np.frombuffer(out, np.float32).flatten()
    else:
        # prepare input data
        audio, source_sr = librosa.load(file, sr=None)
        # Resample the wav if needed
        if source_sr is not None and source_sr != sr:
            audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

    return audio


def tensor_chunk(tensor, offset=0, length=None):
    total_length = (
        tensor.shape[-1] if isinstance(tensor, np.ndarray) else chunk_shape(tensor)[-1]
    )

    if length is None:
        length = total_length - offset
    else:
        length = min(total_length - offset, length)

    if isinstance(tensor, TensorChunk):
        offset = offset + tensor.offset
        tensor = tensor.tensor
    else:
        tensor = tensor
        offset = offset

    return TensorChunk(tensor, offset, length)


def chunk_shape(chunk: TensorChunk):
    shape = list(chunk.tensor.shape)
    shape[-1] = chunk.length
    return shape


def chunk_padded(chunk: TensorChunk, target_length):
    length = chunk_shape(chunk)[-1]
    delta = target_length - length
    total_length = chunk.tensor.shape[-1]

    start = chunk.offset - delta // 2
    end = start + target_length

    correct_start = max(0, start)
    correct_end = min(total_length, end)

    pad_left = correct_start - start
    pad_right = end - correct_end

    out = np.pad(
        chunk.tensor[..., correct_start:correct_end],
        ((0, 0), (0, 0), (pad_left, pad_right)),
    )

    return out


# ======================
# Main functions
# ======================


def apply_model(
    net,
    mix,
    shifts: int = 1,
    split: bool = True,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    segment: Optional[float] = None,
):
    samplerate = 44100

    batch, channels, length = (
        mix.shape if isinstance(mix, np.ndarray) else chunk_shape(mix)
    )
    if shifts:
        max_shift = int(0.5 * samplerate)
        mix = tensor_chunk(mix)
        padded_mix = chunk_padded(mix, length + 2 * max_shift)
        out = 0.0
        for shift_idx in range(shifts):
            offset = np.random.randint(0, max_shift)
            shifted = tensor_chunk(padded_mix, offset, length + max_shift - offset)
            res = apply_model(
                net,
                shifted,
                shifts=0,
                split=split,
                overlap=overlap,
                transition_power=transition_power,
            )
            shifted_out = res
            out += shifted_out[..., max_shift - offset :]

        out /= shifts
        return out
    elif split:
        out = np.zeros((batch, 4, channels, length))
        sum_weight = np.zeros(length)
        if segment is None:
            segment = Fraction(39, 5)

        segment_length = int(samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = range(0, length, stride)
        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment. Then we normalize and take to the power `transition_power`.
        # Large values of transition power will lead to sharper transitions.
        weight = np.concatenate(
            [
                np.arange(1, segment_length // 2 + 1),
                np.arange(segment_length - segment_length // 2, 0, -1),
            ]
        )
        # If the overlap < 50%, this will translate to linear transition when
        # transition_power is 1.
        weight = (weight / weight.max()) ** transition_power
        for offset in offsets:
            chunk = tensor_chunk(mix, offset, segment_length)
            chunk_out = apply_model(
                net,
                chunk,
                shifts=0,
                split=False,
                overlap=overlap,
                transition_power=transition_power,
            )
            chunk_length = chunk_out.shape[-1]
            out[..., offset : offset + segment_length] += (
                weight[:chunk_length] * chunk_out
            )
            sum_weight[offset : offset + segment_length] += weight[:chunk_length]
            offset += segment_length

        out /= sum_weight
        return out
    else:
        valid_length = length
        # if isinstance(model, HTDemucs) and segment is not None:
        #     valid_length = int(segment * model.samplerate)
        # elif hasattr(model, "valid_length"):
        #     valid_length = model.valid_length(length)  # type: ignore
        mix = tensor_chunk(mix)
        padded_mix = chunk_padded(mix, valid_length)

        # apply the model
        if not args.onnx:
            output = net.predict([padded_mix])
        else:
            output = net.run(None, {"mix": padded_mix})
        output = output[0]

        delta = output.shape[-1] - length
        if delta:
            output = output[..., delta // 2 : -(delta - delta // 2)]

    return output


def predict(net, audio):
    wav = np.load("wav.npy")
    # padded_mix = np.load("padded_mix.npy")

    ref = wav.mean(axis=0)
    wav -= ref.mean()
    wav /= ref.std() + 1e-8
    mix = wav[None, :]
    audio_length = wav.shape[1]

    shifts = 1
    split = True
    overlap = 0.25
    transition_power = 1.0

    out = apply_model(net, mix, shifts, split, overlap, transition_power)

    if not args.onnx:
        output = net.predict([padded_mix])
    else:
        output = net.run(None, {"mix": padded_mix})
    out = output[0]

    length = 343980
    delta = out.shape[-1] - length
    if delta:
        out = out[..., delta // 2 : -(delta - delta // 2)]

    chunk_length = chunk_out.shape[-1]
    out[..., offset : offset + segment_length] += weight[:chunk_length] * chunk_out
    sum_weight[offset : offset + segment_length] += weight[:chunk_length].to(mix.device)

    return out


def recognize_from_audio(net):
    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        audio = load_audio(audio_path, SAMPLE_RATE)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            start = int(round(time.time() * 1000))
            output = predict(net, audio)
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"\ttotal processing time {estimation_time} ms")
        else:
            output = predict(net, audio)

        # # save result
        # savepath = get_savepath(args.savepath, audio_path, ext=".wav")
        # logger.info(f"saved at : {savepath}")
        # sf.write(savepath, output, sr)

    logger.info("Script finished successfully.")


def main():
    WEIGHT_PATH = args.model_file
    MODEL_PATH = WEIGHT_PATH.replace(".onnx", ".onnx.prototxt")
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime

        providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    recognize_from_audio(net)


if __name__ == "__main__":
    main()
