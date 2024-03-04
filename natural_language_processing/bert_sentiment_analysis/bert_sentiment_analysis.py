import time
import sys

from transformers import DistilBertTokenizer
import numpy

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

DEFAULT_TEXT = 'Transformers and ailia SDK is an awesome combo!'
# DEFAULT_TEXT = "I'm sick today."


parser = get_base_parser('bert sentiment-analysis.', None, None)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='TEXT', default=DEFAULT_TEXT,
    help='input text'
)
args = update_parser(parser, check_input_type=False)


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = "distilbert-base-uncased-finetuned-sst-2-english.onnx"
MODEL_PATH = "distilbert-base-uncased-finetuned-sst-2-english.onnx.prototxt"
REMOTE_PATH = \
    "https://storage.googleapis.com/ailia-models/bert_sentiment_analysis/"


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    tokenizer = DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased-finetuned-sst-2-english'
    )
    model_inputs = tokenizer.encode_plus(args.input, return_tensors="pt")
    inputs_onnx = {
        k: v.cpu().detach().numpy() for k, v in model_inputs.items()
    }

    logger.info("Input : "+str(args.input))

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            score = ailia_model.predict(inputs_onnx)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        score = ailia_model.predict(inputs_onnx)

    score = numpy.exp(score) / numpy.exp(score).sum(-1, keepdims=True)

    label_name = ["negative", "positive"]

    label_id = numpy.argmax(numpy.array(score))
    logger.info("Label : "+str(label_name[label_id]))
    logger.info("Score : "+str(score[0][0][label_id]))

    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
