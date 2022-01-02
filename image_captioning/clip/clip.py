import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from math_utils import softmax  # noqa: E402C
# logger
from logging import getLogger  # noqa: E402

from simple_tokenizer import SimpleTokenizer as _Tokenizer

logger = getLogger(__name__)

_tokenizer = _Tokenizer()

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'ViT-B32.onnx'
MODEL_PATH = 'ViT-B32.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/clip/'

IMAGE_PATH = 'chelsea.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'CLIP', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-t', '--text', dest='text_inputs', type=str,
    action='append',
    help='Input text. (can be specified multiple times)'
)
parser.add_argument(
    '--desc_file', default=None, metavar='DESC_FILE', type=str,
    help='description file'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def tokenize(texts, context_length=77, truncate=False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

        result[i, :len(tokens)] = np.array(tokens)

    return result


def preprocess(img):
    h, w = (IMAGE_SIZE, IMAGE_SIZE)
    im_h, im_w, _ = img.shape

    # resize
    scale = h / min(im_h, im_w)
    ow, oh = int(im_w * scale), int(im_h * scale)
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.BICUBIC))

    # center_crop
    if ow > w:
        x = (ow - w) // 2
        img = img[:, x:x + w, :]
    if oh > h:
        y = (oh - h) // 2
        img = img[y:y + h, :, :]

    img = img[:, :, ::-1]  # BGR -> RBG
    img = img / 255

    mean = np.array((0.48145466, 0.4578275, 0.40821073))
    std = np.array((0.26862954, 0.26130258, 0.27577711))
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def predict(net, img, text):
    img = preprocess(img)

    text = tokenize(text)

    # feedforward
    if not args.onnx:
        output = net.predict([img, text])
    else:
        output = net.run(None, {'image': img, 'text': text})

    logits_per_image, logits_per_text = output

    pred = softmax(logits_per_image, axis=1)

    return pred[0]


def recognize_from_image(net):
    text_inputs = args.text_inputs
    desc_file = args.desc_file
    if desc_file:
        with open(desc_file) as f:
            text_inputs = [x.strip() for x in f.readlines() if x.strip()]
    elif text_inputs is None:
        text_inputs = [f"a {c}" for c in ("human", "dog", "cat")]

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                pred = predict(net, img, text_inputs)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net, img, text_inputs)

    inds = np.argsort(-pred)[:5]
    logger.info("Top predictions:\n")
    for i in inds:
        logger.info(f"{text_inputs[i]:>16s}: {100 * pred[i]:.2f}%")

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
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
