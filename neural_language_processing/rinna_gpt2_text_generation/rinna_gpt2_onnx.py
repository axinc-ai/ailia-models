import time
import sys
import os
import onnxruntime as rt
from transformers import T5Tokenizer
import numpy

from utils_rinna_gpt2 import *
import ailia

sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

onnx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rinna_gpt2')
os.chdir(onnx_dir)

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
    '--outlength', '-c', default=30
)
args = update_parser(parser, check_input_type=False)


ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider",]

# ======================
# OPTIMIZATIONS
# ======================
opt = rt.SessionOptions()
opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
opt.log_severity_level = 4
opt.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = "japanese-gpt2-small.onnx"
MODEL_PATH = "rinna_gpt2/japanese-gpt2-small.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/rinna_gpt2_text_generation/"


# ======================
# Main function
# ======================
def main():
    #ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    ailia_model = rt.InferenceSession(
    WEIGHT_PATH,
    opt
)
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-small")
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
    #check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    main()
