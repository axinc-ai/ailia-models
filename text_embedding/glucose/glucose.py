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

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'GLuCoSE-base-ja.onnx'
MODEL_PATH = 'GLuCoSE-base-ja.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/glucose/'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'GLuCoSE-base-Japanese', None, None
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

# def draw_bbox(img, bboxes):
#     return img


# ======================
# Main functions
# ======================


def predict(models, sentences):
    length_sorted_idx = [0, 2, 1]
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

    tokenizer = models["tokenizer"]

    batch_size = 32
    for start_index in range(0, len(sentences), batch_size):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]

        to_tokenize = [sentences_batch]
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        max_seq_length = 512
        features = tokenizer(
            *to_tokenize,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=max_seq_length,
        )
        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]

    return


def recognize_from_sentence(models):
    sentences = [
        "PKSHA Technologyは機械学習/深層学習技術に関わるアルゴリズムソリューションを展開している。",
        "この深層学習モデルはPKSHA Technologyによって学習され、公開された。",
        "広目天は、仏教における四天王の一尊であり、サンスクリット語の「種々の眼をした者」を名前の由来とする。",
    ]

    logger.info(sentences)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            out = predict(net, img)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Logging
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        out = predict(models, sentences)

    logger.info('Script finished successfully.')


def main():
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

    tokenizer = AutoTokenizer.from_pretrained("tokenizer")

    models = {
        "net": net,
        "tokenizer": tokenizer,
    }

    recognize_from_sentence(models)


if __name__ == '__main__':
    main()
