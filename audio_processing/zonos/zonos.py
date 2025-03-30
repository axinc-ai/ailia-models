import sys
import time
from logging import getLogger

import librosa
import numpy as np

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_EMB_PATH = "speaker_embedding.onnx"
WEIGHT_COND_PATH = "conditioner.onnx"
WEIGHT_FIRST_PATH = "generator_first.onnx"
WEIGHT_SECOND_PATH = "generator_second.onnx"
WEIGHT_DEC_PATH = "autoencoder.onnx"
MODEL_EMB_PATH = "speaker_embedding.onnx.prototxt"
MODEL_COND_PATH = "conditioner.onnx.prototxt"
MODEL_FIRST_PATH = "generator_first.onnx.prototxt"
MODEL_SECOND_PATH = "generator_second.onnx.prototxt"
MODEL_DEC_PATH = "autoencoder.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/zonos/"

SAVE_WAV_PATH = "output.wav"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Zonos", None, SAVE_WAV_PATH)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


class EspeakPhonemeConditioner:
    def forward(self, x):
        return x


def prefix_conditioner(net, cond_dict):
    if not args.onnx:
        output = net.run(
            {
                "espeak": cond_dict["espeak"],
                "speaker": cond_dict["speaker"],
                "emotion": cond_dict["emotion"],
                "fmax": cond_dict["fmax"],
                "pitch_std": cond_dict["pitch_std"],
                "speaking_rate": cond_dict["speaking_rate"],
                "language_id": cond_dict["language_id"],
            },
        )
    else:
        output = net.run(
            None,
            {
                "espeak": cond_dict["espeak"],
                "speaker": cond_dict["speaker"],
                "emotion": cond_dict["emotion"],
                "fmax": cond_dict["fmax"],
                "pitch_std": cond_dict["pitch_std"],
                "speaking_rate": cond_dict["speaking_rate"],
                "language_id": cond_dict["language_id"],
            },
        )
    conditioning = output[0]

    return conditioning


# ======================
# Main functions
# ======================


def generate_voice(models):
    ref_audio, sr = librosa.load(input_audio, sr=16_000)

    net = models["embedding"]
    if not args.onnx:
        output = net.run({"wav": wav})
    else:
        output = net.run(None, {"wav": wav})
    speaker = output[0]

    espeak_cond = EspeakPhonemeConditioner().forward((["Hello, world!"], ["en-us"]))

    cond_dict = dict(
        espeak=espeak_cond,
        speaker=speaker,
        emotion=np.array(
            [0.3078, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2565, 0.3078],
            dtype=np.float32,
        ).reshape(1, 1, -1),
        fmax=np.array([22050], dtype=np.float32).reshape(1, 1, -1),
        pitch_std=np.array([20], dtype=np.float32).reshape(1, 1, -1),
        speaking_rate=np.array([15], dtype=np.float32).reshape(1, 1, -1),
        language_id=np.array([24], dtype=int).reshape(1, 1, -1),
    )
    uncond_dict = dict(
        espeak=espeak_cond,
        speaker=speaker[:, :0, :],
        emotion=np.zeros((1, 0, 8), dtype=np.float32),
        fmax=np.zeros((1, 0, 1), dtype=np.float32),
        pitch_std=np.zeros((1, 0, 1), dtype=np.float32),
        speaking_rate=np.zeros((1, 0, 1), dtype=np.float32),
        language_id=np.zeros((1, 0, 1), dtype=int),
    )

    net = models["conditioner"]
    conditioning = np.concatenate(
        [prefix_conditioner(net, cond_dict), prefix_conditioner(net, uncond_dict)]
    )

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_EMB_PATH, MODEL_EMB_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_COND_PATH, MODEL_COND_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_FIRST_PATH, MODEL_FIRST_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_SECOND_PATH, MODEL_SECOND_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        embedding = ailia.Net(MODEL_EMB_PATH, WEIGHT_EMB_PATH, env_id=env_id)
        conditioner = ailia.Net(MODEL_COND_PATH, WEIGHT_COND_PATH, env_id=env_id)
        first_net = ailia.Net(MODEL_FIRST_PATH, WEIGHT_FIRST_PATH, env_id=env_id)
        second_net = ailia.Net(MODEL_SECOND_PATH, WEIGHT_SECOND_PATH, env_id=env_id)
        decoder = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        embedding = onnxruntime.InferenceSession(WEIGHT_EMB_PATH, providers=providers)
        conditioner = onnxruntime.InferenceSession(
            WEIGHT_COND_PATH, providers=providers
        )
        first_net = onnxruntime.InferenceSession(WEIGHT_FIRST_PATH, providers=providers)
        second_net = onnxruntime.InferenceSession(
            WEIGHT_SECOND_PATH, providers=providers
        )
        decoder = onnxruntime.InferenceSession(WEIGHT_DEC_PATH, providers=providers)

    models = {
        "embedding": embedding,
        "conditioner": conditioner,
        "first_net": first_net,
        "second_net": second_net,
        "decoder": decoder,
    }

    generate_voice(models)


if __name__ == "__main__":
    main()
