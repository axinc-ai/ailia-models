import sys
import time
from logging import getLogger

from transformers import AutoTokenizer
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

import pprint
from scipy.special import softmax

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'bert_ner_ja.onnx'
MODEL_PATH = 'bert_ner_ja.onnx.prototxt'

id2label = {
  0: "O",
  1: "B-人名",
  2: "I-人名",
  3: "B-法人名",
  4: "I-法人名",
  5: "B-政治的組織名",
  6: "I-政治的組織名",
  7: "B-その他の組織名",
  8: "I-その他の組織名",
  9: "B-地名",
  10: "I-地名",
  11: "B-施設名",
  12: "I-施設名",
  13: "B-製品名",
  14: "I-製品名",
  15: "B-イベント名",
  16: "I-イベント名"
}

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_ner_ja/"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'named entity recognition using bert-japanese', None, None
)

parser.add_argument(
    "-f", "--file", metavar="PATH", type=str,
    default="input.txt",
    help="Input text file path."
)

parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default=None,
    help="Input text."
)

args = update_parser(parser, check_input_type=False)

# ======================
# Helper functions
# ======================

def handle_subwords(token):
    r"""
    Description:
        Get rid of subwords '##'.
    About tokenizer subwords:
        See: https://huggingface.co/docs/transformers/tokenizer_summary
    """
    if len(token) > 2 and token[0:2] == '##':
        token = token[2:]
    return token


# ======================
# Main functions
# ======================


def recognize(model):
    tokenizer = model['tokenizer']
    model = model['bert']

    input_text = args.input
    input_path = args.file
    if input_text is None:
        input_text = open(input_path, "r", encoding="utf-8").read()
    encoded = tokenizer(input_text, max_length = 512, return_tensors='np')
    input_ids, attention_mask = encoded['input_ids'], encoded['attention_mask']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][1:-1])# remove special tokens
    tokens = [handle_subwords(token) for token in tokens]

    logger.info("input_text: %s" % input_text)

    # inference
    logger.info('inference has started...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))

            out = model.predict((input_ids, attention_mask))[0]

            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Logging
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        #prediction = predict(model, input_text)
        out = model.predict((input_ids, attention_mask))[0]
    

    out = out[0, 1:-1]# remove special tokens
    out = softmax(out, axis=-1)
    label_ids = np.argmax(out, axis=-1)
    labels = [id2label[i] for i in label_ids]
    result = {i: {'token': token, 'label': label, 'score': out[i, label_ids[i]]} for i, (token, label) in enumerate(zip(tokens, labels))}
    
    logger.info('predicted entities:')
    logger.info(pprint.pformat(result))

    # save output
    if args.savepath is not None:
        import pickle
        with open(args.savepath, mode="wb") as f:
            pickle.dump(result, f)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    model_name = "jurabi/bert-ner-japanese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    env_id = args.env_id

    bert = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id = env_id)
    # initialize
    model = {
        'bert':bert,
        'tokenizer':tokenizer
    }

    recognize(model)

if __name__ == '__main__':
    main()