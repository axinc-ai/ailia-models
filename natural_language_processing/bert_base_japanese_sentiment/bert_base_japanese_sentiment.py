import sys
import time

import numpy as np
import cv2
from PIL import Image

from transformers import AutoTokenizer

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "model.onnx"
MODEL_PATH = "model.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/bert_base_japanese_sentiment/'

DEFAULT_TEXT = "私は幸せである。"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'bert-base-japanese-sentiment', None, None
)
parser.add_argument(
    '--input', '-i', default=DEFAULT_TEXT
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Secondaty Functions
# ======================

# ======================
# Main functions
# ======================

def preprocess(tokenizer, sequence):
    encoded_sequence = tokenizer.encode(sequence)
    encoded_sequence = np.array(encoded_sequence)

    encoded_sequence = np.expand_dims(encoded_sequence, axis=0)
    encoded_sequence = encoded_sequence.astype(np.int64)

    return encoded_sequence


def post_processing(classifier, output):
    output = classifier["weight"] @ output
    output = output + classifier["bias"]

    return output


def predict(model_info, sequence):
    tokenizer = model_info["tokenizer"]
    net = model_info["model"]
    classifier = model_info["classifier"]

    input_ids = preprocess(tokenizer, sequence)
    attention_mask = np.ones(input_ids.shape, dtype=np.int64)
    token_type_ids = np.zeros(input_ids.shape, dtype=np.int64)

    # feedforward
    if not args.onnx:
        output = net.predict([input_ids, attention_mask, token_type_ids])
    else:
        output = net.run(None, {
            'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids
        })

    last_hidden_state, pooled_output = output

    logits = post_processing(classifier, pooled_output[0])

    return logits


def recognize_from_text(model_info):
    # prepare input data
    sequence = args.input

    logger.info("Input : " + sequence)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            logits = predict(model_info, sequence)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Logging
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        logits = predict(model_info, sequence)

    # # plot result
    # savepath = get_savepath(args.savepath, image_path, ext='.png')
    # logger.info(f'saved at : {savepath}')
    # cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # initialize
    tokenizer = AutoTokenizer.from_pretrained("daigo/bert-base-japanese-sentiment")

    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    classifier = np.load("classifier.npy", allow_pickle=True).item()

    model_info = {
        "tokenizer": tokenizer,
        "model": net,
        "classifier": classifier,
    }

    recognize_from_text(model_info)


if __name__ == '__main__':
    main()
