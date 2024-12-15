import time
import sys
import os
from transformers import AutoTokenizer
import numpy

from utils_phi3 import *
import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Arguemnt Parser Config
# ======================

DEFAULT_TEXT = '<|user|>Tell me a joke<|end|><|assistant|>'

parser = get_base_parser('phi3', None, None)
# overwrite
parser.add_argument(
    '--input', '-i', default=DEFAULT_TEXT
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime'
)
args = update_parser(parser, check_input_type=False)


# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = "phi3-mini-128k-instruct-cuda-fp16.onnx"
MODEL_PATH = None
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/phi3/"


# ======================
# Main function
# ======================
def main():
    if args.onnx:
        import onnxruntime
        ailia_model = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        logger.info("This model requires multiple input shape, so running on CPU")
        memory_mode = ailia.get_memory_mode(True, True, False, True)
        ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, memory_mode = memory_mode, env_id=0)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    logger.info("Input : "+args.input)

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            output = generate_text(tokenizer, ailia_model, args.input, args.onnx)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        output = generate_text(tokenizer, ailia_model, args.input, args.onnx)

    logger.info("output : "+output)
    logger.info('Script finished successfully.')


if __name__ == "__main__":
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    main()
