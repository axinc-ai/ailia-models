import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'landmarks_classifier_asia_V1_1.onnx'
MODEL_PATH = 'landmarks_classifier_asia_V1_1.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/landmarks_classifier_asia/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 321

LABEL_MAP_FILE = 'landmarks_classifier_asia_V1_label_map.csv'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'landmarks_classifier_asia', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-n', '--top_k', default=20, type=int,
    help='Use object detection.'
)
parser.add_argument(
    '-k', '--keep_ratio',
    action='store_true',
    help='keep ratio on resize.'
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

def read_label_map():
    label_map = {}
    label_set = {}
    with open(LABEL_MAP_FILE, encoding='utf-8') as f:
        for i, row in enumerate(f):
            if i == 0:
                continue
            row = row.strip()
            _id, name = row.split(',', 1)
            if 2 < len(name) and name[0] == '"' and name[-1] == '"':
                name = name[1:-1]

            _id = int(_id)
            label_map[_id] = name

            ids = label_set[name] = label_set.get(name, [])
            ids.append(_id)

    return label_map, label_set


# ======================
# Main functions
# ======================

def preprocess(img):
    keep_ratio = args.keep_ratio

    im_h, im_w, _ = img.shape

    if keep_ratio:
        scale = IMAGE_SIZE / max(im_h, im_w)
        ow, oh = int(im_w * scale), int(im_h * scale)
        if ow != im_w or oh != im_h:
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

        if ow != IMAGE_SIZE or oh != IMAGE_SIZE:
            pad = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
            pad_h, pad_w = pad.shape[:2]
            pad_h = (pad_h - oh) // 2
            pad_w = (pad_w - ow) // 2
            pad[pad_h:pad_h + oh, pad_w:pad_w + ow] = img
            img = pad
    else:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    img = img / 255
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(logits, label_map, label_set):
    top_k = args.top_k

    upd_logits = np.ones_like(logits) * -1e+3
    for name, ids in label_set.items():
        i = np.argmax(logits[ids])
        _id = ids[i]
        upd_logits[_id] = logits[_id]

    idx = np.argsort(-upd_logits)
    idx = idx[:top_k]

    pred = []
    for i in idx:
        name = label_map[i]
        pred.append((name, upd_logits[i]))

    return pred


def predict(net, img):
    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'keras_layer_input': img})

    logits = output[0]

    return logits[0]


def recognize_from_image(net):
    label_map, label_set = read_label_map()

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
                logits = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            logits = predict(net, img)

        pred = post_processing(logits, label_map, label_set)

        logger.info("Top predictions:")
        for x in pred:
            logger.info(f"  {x[0]}: {100 * x[1]:.2f}%")

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
