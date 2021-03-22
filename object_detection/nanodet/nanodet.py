# ailia predict api sample

import numpy as np
import time
import os
import sys
import cv2
import torch

from lib_nanodet.transform import pipeline
from lib_nanodet.config import cfg, load_config
from lib_nanodet.nanodet_head import NanoDetHead
from lib_nanodet import visualize

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models
from detector_utils import plot_results, write_predictions, load_image
import webcamera_utils

# logger
from logging import getLogger

logger = getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/nanodet/'

# settings
WEIGHT_PATH = "nanodet-EfficientNet-Lite2_512.opt.onnx"
MODEL_PATH = "nanodet-EfficientNet-Lite2_512.opt.onnx.prototxt"

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

# Default input size
HEIGHT = 512
WIDTH = 512

model_conf_path = './config_nanodet/EfficientNet-Lite/nanodet-EfficientNet-Lite2_512.yml'
load_config(cfg, model_conf_path)

class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
    'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
    'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
    'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
]

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('nanodet model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)


# ======================
# Detection function
# ======================
def detection(raw_img, filename, net):
    img_info = {}
    img_info['file_name'] = filename

    height, width = raw_img.shape[:2]
    img_info['height'] = height
    img_info['width'] = width
    meta = dict(img_info=img_info,
                raw_img=raw_img,
                img=raw_img)

    pipe = pipeline.Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    meta = pipe(meta, cfg.data.val.input_size)

    img = cv2.resize(raw_img, (HEIGHT, WIDTH))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    meta['img'] = img

    trans_detections = []

    detections = net.run(img)
    for detection in detections:
        trans_detections.append(detection.transpose(0, 2, 1))

    cls_scores1, cls_scores2, cls_scores3 = trans_detections[:3]
    bbox_preds1, bbox_preds2, bbox_preds3 = trans_detections[3:]

    # ONNXエクスポートの際に予測結果の形状が変化するため修正
    # → (nanodet_head.py:108行)の処理による影響
    cls_scores1 = cls_scores1.reshape(1, 80, 64, 64)
    cls_scores2 = cls_scores2.reshape(1, 80, 32, 32)
    cls_scores3 = cls_scores3.reshape(1, 80, 16, 16)

    bbox_preds1 = bbox_preds1.reshape(1, 44, 64, 64)
    bbox_preds2 = bbox_preds2.reshape(1, 44, 32, 32)
    bbox_preds3 = bbox_preds3.reshape(1, 44, 16, 16)

    # 動作確認のためnumpy→torchへ変換する
    cls_scores1 = torch.from_numpy(cls_scores1).clone()
    cls_scores2 = torch.from_numpy(cls_scores2).clone()
    cls_scores3 = torch.from_numpy(cls_scores3).clone()

    bbox_preds1 = torch.from_numpy(bbox_preds1).clone()
    bbox_preds2 = torch.from_numpy(bbox_preds2).clone()
    bbox_preds3 = torch.from_numpy(bbox_preds3).clone()

    return (cls_scores1, cls_scores2, cls_scores3), (bbox_preds1, bbox_preds2, bbox_preds3), meta


# ======================
# Main functions
# ======================
def recognize_from_image():
    env_id = ailia.get_gpu_environment_id()
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    # net.set_input_shape(WIDTH, HEIGHT)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = cv2.imread(image_path)
        logger.debug(f'input image shape: {raw_img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                detection(raw_img, IMAGE_PATH, net)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            cls_scores, bbox_preds, meta = detection(raw_img, IMAGE_PATH, net)
            preds = cls_scores, bbox_preds
            cfg_head = cfg.model.arch.head
            head = NanoDetHead(**cfg_head)
            results = head.post_process(preds, meta)
            result = visualize.show_result(raw_img, results, class_names)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, result)

    if cv2.waitKey(0) != 32:  # space bar
        exit()


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape(WIDTH, HEIGHT)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = f_h, f_w
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        raw_img = frame
        cls_scores, bbox_preds, meta = detection(raw_img, IMAGE_PATH, net)
        preds = cls_scores, bbox_preds
        cfg_head = cfg.model.arch.head
        head = NanoDetHead(**cfg_head)
        results = head.post_process(preds, meta)
        result = visualize.show_result(raw_img, results, class_names)

        # save results
        if writer is not None:
            writer.write(result)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    # check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
