import time
import sys
from transformers import AutoTokenizer

from utils_gpt2 import *
import ailia

sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Arguemnt Parser Config
# ======================

DEFAULT_TEXT = 'My name is Clara and I am'

parser = get_base_parser('gpt2 text generation', None, None)
# overwrite
parser.add_argument(
    '--input', '-i', default=DEFAULT_TEXT
)
parser.add_argument(
    '--outlength', '-o', default=30
)
args = update_parser(parser, check_input_type=False)


ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider",]

# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = "gpt2-medium.onnx"
MODEL_PATH = "gpt2-medium.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/gpt2/"


# ======================
# Main function
# ======================
def main():
    ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    logger.info("Input : "+args.input)

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            output = generate_text(tokenizer, ailia_model, args.input, args.outlength)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        output = generate_text(tokenizer, ailia_model, args.input, args.outlength)

    logger.info("output : "+output)
    logger.info('Script finished successfully.')


if __name__ == "__main__":
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    main()
