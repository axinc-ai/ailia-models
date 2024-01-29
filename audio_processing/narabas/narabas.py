import sys
import time
from typing import Union

import ailia  # noqa: E402
import numpy as np
import onnxruntime

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from symbols import BOS, EOS, phoneme_to_id

logger = getLogger(__name__)

NARABAS_WEIGHT_PASS = "narabas.onnx"
NARABAS_MODEL_PATH = "narabas.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/narabas/"
AUDIO_PATH = "input.wav"

parser = get_base_parser('narabas', AUDIO_PATH, None)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime'
)
parser.add_argument(
    "--phonemes",
    type=str,
    help="phoneme (splitted by space)",
)
parser.add_argument(
    "--sample_rate",
    type=int,
    help="sample rate",
    default=1,
)
args = update_parser(parser)


def create_instance():
    if not args.onnx:
        logger.info("User ailia")
        env_id = args.env_id
        logger.info(f"{env_id: {env_id}}")
        memory_mode = ailia.get_memory_mode(reuse_interstage=True)
        session = ailia.Net(MODEL_PATH, WEIGHGT_PATH, env_id=env_id, memory_mode=memory_mode)
    else:
        logger.info("Use onnxruntime")
        import onnxruntime
        session = onnxruntime.InferenceSession(WEIGHGT_PATH)

    return session


def execute_session(session, wav):
    if not args.onnx:
        result = session.run(wav)[0]
    else:
        result = session.run(
            ["output"],
            {"input": wav.numpy()}
        )

    return result


def infer(net: Union[ailia.Net, onnxruntime.InferenceSession]):
    input_audio_filename = args.input
    sample_rate = args.sample_rate
    phonemes = args.phonemes
    wav = load_audio(input_audio_filename, sample_rate)
    phn_ids = [phoneme_to_id[phn] for phn in phonemes.split()]
    phn_ids = [BOS, *phn_ids, EOS]

    session = create_instance()


if __name__ == "__main__":
    check_and_download_models(NARABAS_WEIGHT_PASS, NARABAS_MODEL_PATH, REMOTE_PATH)
    # net initialize
    if not args.onnx:
        net = ailia.Net(NARABAS_MODEL_PATH, NARABAS_WEIGHT_PASS, env_id=args.env_id)
    else:
        net = onnxruntime.InferenceSession(NARABAS_WEIGHT_PASS)
