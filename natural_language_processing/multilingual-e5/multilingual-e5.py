import sys
import time
from logging import getLogger

import numpy as np
from transformers import AutoTokenizer

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'xxx.onnx'
MODEL_PATH = 'xxx.onnx.prototxt'
WEIGHT_XXX_PATH = 'xxx.onnx'
MODEL_XXX_PATH = 'xxx.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/multilingual-e5/'

SENTENCE_PATH = 'sample.txt'
MIN_LENGTH = 5

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Multilingual-E5', SENTENCE_PATH, None
)
parser.add_argument(
    '-p', '--prompt', metavar='PROMPT', default=None,
    help='Specify input prompt. If not specified, script runs interactively.'
)
parser.add_argument(
    '-m', '--model_type', default='base', choices=('base', 'large'),
    help='model type'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def read_sentences(file_path):
    if file_path.endswith(".pdf"):
        from pdfminer.high_level import extract_text
        text = extract_text(file_path)
    else:
        with open(file_path, "r") as f:
            text = f.read()

    sents = text.replace('\n', '').split('。')
    sents = [s.strip() + '。' for s in sents if len(s.strip()) > MIN_LENGTH]

    return sents


def average_pool(
        last_hidden_states,
        attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# ======================
# Main functions
# ======================

def predict(models, sentences):
    input_texts = ['query: {}'.format(t) for t in sentences]

    tokenizer = models['tokenizer']
    batch_dict = tokenizer(
        input_texts,
        max_length=512, padding=True, truncation=True,
        return_tensors='np')

    net = models['net']

    # feedforward
    if not args.onnx:
        output = net.predict([batch_dict['input_ids'], batch_dict['attention_mask']])
    else:
        output = net.run(None, {
            "input_ids": batch_dict["input_ids"],
            "attention_mask": batch_dict["attention_mask"]
        })
    last_hidden_state = output[0]

    embeddings = average_pool(last_hidden_state, batch_dict['attention_mask'])

    return embeddings


def recognize_from_sentence(models):
    # extract sentences to list
    sentences = read_sentences(args.input[0])

    # inference
    logger.info("Generating embeddings...")
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            features, attn_mask = predict(models, sentences)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
        exit()
    else:
        features, attn_mask = predict(models, sentences)

    logger.info('Script finished successfully.')


def main():
    # dic_model = {
    #     'base': (WEIGHT_PATH, MODEL_PATH),
    #     'large': (WEIGHT_XXX_PATH, MODEL_XXX_PATH),
    # }
    # WEIGHT_PATH, MODEL_PATH = dic_model[args.model_type]
    WEIGHT_PATH = "multilingual-e5-base.onnx"
    MODEL_PATH = "multilingual-e5-base.onnx.prototxt"

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    tokenizer = AutoTokenizer.from_pretrained('tokenizer')

    models = {
        "net": net,
        "tokenizer": tokenizer,
    }

    recognize_from_sentence(models)


if __name__ == '__main__':
    main()
