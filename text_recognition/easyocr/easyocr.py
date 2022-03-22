import sys
import time

import numpy as np
import cv2

import ailia
# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# for EasyOCR
from easyocr_utils import *


# ======================
# PARAMETERS
# ======================
DETECTOR_MODEL_PATH = 'detector_craft.onnx.prototxt'
DETECTOR_WEIGHT_PATH = 'detector_craft.onnx'

RECOGNIZER_CHINESE_MODEL_PATH = 'recognizer_zh_sim_g2.onnx.prototxt'
RECOGNIZER_CHINESE_WEIGHT_PATH = 'recognizer_zh_sim_g2.onnx'

RECOGNIZER_JAPANESE_MODEL_PATH = 'recognizer_japanese_g2.onnx.prototxt'
RECOGNIZER_JAPANESE_WEIGHT_PATH = 'recognizer_japanese_g2.onnx'

RECOGNIZER_ENGLISH_MODEL_PATH = 'recognizer_english_g2.onnx.prototxt'
RECOGNIZER_ENGLISH_WEIGHT_PATH = 'recognizer_english_g2.onnx'

RECOGNIZER_FRENCH_MODEL_PATH = 'recognizer_latin_g2.onnx.prototxt'
RECOGNIZER_FRENCH_WEIGHT_PATH = 'recognizer_latin_g2.onnx'

RECOGNIZER_KOREAN_MODEL_PATH = 'recognizer_korean_g2.onnx.prototxt'
RECOGNIZER_KOREAN_WEIGHT_PATH = 'recognizer_korean_g2.onnx'

RECOGNIZER_THAI_MODEL_PATH = 'recognizer_thai.onnx.prototxt'
RECOGNIZER_THAI_WEIGHT_PATH = 'recognizer_thai.onnx'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/easyocr/'

IMAGE_PATH = 'example/chinese.jpg'
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Ready-to-use OCR', IMAGE_PATH, None
)
parser.add_argument(
    '-l', '--language', type=str, default='chinese',
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img_grey = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(image_path)

        # predict
        horizontal_list, free_list = detector_predict(detector, img)
        result = recognizer_predict(args.language, lang_list, character, symbol, recognizer, img_grey, horizontal_list[0], free_list[0])

        # show
        logger.info('detect result')
        logger.info('{}, {}'.format(horizontal_list[0], free_list[0]))
        #logger.info('recognize result {}'.format(result))
        logger.info('recognize result')
        for r in result:
            logger.info(r)


if __name__ == '__main__':
    # model files check and download
    check_and_download_models(DETECTOR_WEIGHT_PATH, DETECTOR_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RECOGNIZER_CHINESE_WEIGHT_PATH, RECOGNIZER_CHINESE_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RECOGNIZER_JAPANESE_WEIGHT_PATH, RECOGNIZER_JAPANESE_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RECOGNIZER_ENGLISH_WEIGHT_PATH, RECOGNIZER_ENGLISH_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RECOGNIZER_FRENCH_WEIGHT_PATH, RECOGNIZER_FRENCH_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RECOGNIZER_KOREAN_WEIGHT_PATH, RECOGNIZER_KOREAN_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RECOGNIZER_THAI_WEIGHT_PATH, RECOGNIZER_THAI_MODEL_PATH, REMOTE_PATH)

    # set model
    detector = ailia.Net(DETECTOR_MODEL_PATH, DETECTOR_WEIGHT_PATH, env_id=args.env_id)
    if args.language == 'chinese':
        recognizer = ailia.Net(RECOGNIZER_CHINESE_MODEL_PATH, RECOGNIZER_CHINESE_WEIGHT_PATH, env_id=args.env_id)
        lang_list = ['ch_sim','en']
        character = recognition_models['zh_sim_g2']['characters']
        symbol = recognition_models['zh_sim_g2']['symbols']

    elif args.language == 'japanese':
        recognizer = ailia.Net(RECOGNIZER_JAPANESE_MODEL_PATH, RECOGNIZER_JAPANESE_WEIGHT_PATH, env_id=args.env_id)
        lang_list = ['ja','en']
        character = recognition_models['japanese_g2']['characters']
        symbol = recognition_models['japanese_g2']['symbols']

    elif args.language == 'english':
        recognizer = ailia.Net(RECOGNIZER_ENGLISH_MODEL_PATH, RECOGNIZER_ENGLISH_WEIGHT_PATH, env_id=args.env_id)
        lang_list = ['en']
        character = recognition_models['english_g2']['characters']
        symbol = recognition_models['english_g2']['symbols']

    elif args.language == 'french':
        recognizer = ailia.Net(RECOGNIZER_FRENCH_MODEL_PATH, RECOGNIZER_FRENCH_WEIGHT_PATH, env_id=args.env_id)
        lang_list = ['fr', 'en']
        character = recognition_models['latin_g2']['characters']
        symbol = recognition_models['latin_g2']['symbols']

    elif args.language == 'korean':
        recognizer = ailia.Net(RECOGNIZER_KOREAN_MODEL_PATH, RECOGNIZER_KOREAN_WEIGHT_PATH, env_id=args.env_id)
        lang_list = ['ko', 'en']
        character = recognition_models['korean_g2']['characters']
        symbol = recognition_models['korean_g2']['symbols']

    elif args.language == 'thai':
        recognizer = ailia.Net(RECOGNIZER_THAI_MODEL_PATH, RECOGNIZER_THAI_WEIGHT_PATH, env_id=args.env_id)
        lang_list = ['th']
        character = recognition_models['thai_g1']['characters']
        symbol = recognition_models['thai_g1']['symbols']

    else:
        logger.info('invalid language.')
        exit()

    # predict
    if args.video is not None:
        # video mode
        #recognize_from_video()
        exit()
    else:
        # image mode
        recognize_from_image()
