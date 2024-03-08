import time
import sys
import os
from transformers import T5Tokenizer
import numpy

from utils_rinna_gpt2 import *
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

DEFAULT_TEXT = '生命、宇宙、そして万物についての究極の疑問の答えは'

parser = get_base_parser('rinna-gpt2 text generation', None, None)
# overwrite
parser.add_argument(
    '--input', '-i', default=DEFAULT_TEXT
)
parser.add_argument(
    '--outlength', '-o', default=50
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
WEIGHT_PATH = "japanese-gpt2-small.opt.onnx"
MODEL_PATH = "japanese-gpt2-small.opt.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/rinna_gpt2/"


# ======================
# Main function
# ======================
def main():
    if args.onnx:
        import onnxruntime
        ailia_model = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        logger.info("This model requires multiple input shape, so running on CPU")
        ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=0)#args.env_id)
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-small")
    logger.info("Input : "+args.input)

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            output = generate_text(tokenizer, ailia_model, args.input, int(args.outlength), args.onnx)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        output = generate_text(tokenizer, ailia_model, args.input, int(args.outlength), args.onnx)

    logger.info("output : "+output)
    logger.info('Script finished successfully.')


if __name__ == "__main__":
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    main()
