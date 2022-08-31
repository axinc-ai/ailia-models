import argparse
import json
import os
import sys
import time

import ailia
import cv2
import numpy as np

from pytorch_deepfepe_utils import *

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from detector_utils import load_image  # noqa: E402C
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_INPUT_WEIGHTS_PATH = 'ErrorEstimator_input_weights.onnx'
WEIGHT_UPDATE_WEIGHTS_PATH = 'ErrorEstimator_update_weights.onnx'
MODEL_INPUT_WEIGHTS_PATH = 'ErrorEstimator_input_weights.onnx.prototxt'
MODEL_UPDATE_WEIGHTS_PATH = 'ErrorEstimator_update_weights.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pytorch-deepfepe/'
IMAGE_SRC_PATH = 'input_src.jpg'
IMAGE_TGT_PATH = 'input_tgt.jpg'
SAVE_IMAGE_PATH = 'output.png'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'pytorch-deepFEPE', IMAGE_SRC_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser, large_model=True)


# ======================
# Main functions
# ======================

def recognize_from_image(net):
    # Prepare dataset
    dataset = Dataset()
    sample = dataset.prepare()

    # Extract features
    extractor = FeaturesExtractor(if_SP=False)
    matches_use_ori, quality_use = extractor.extract(sample["matches_good"])
    pts_normalized_in, pts1, pts2, T1, T2 = extractor.get_input(matches_use_ori, quality_use)

    # Predict
    _, _, F_est, _, T1, T2, out_layers, _, _, weights, _, _ = \
    net.predict(pts_normalized_in, pts1, pts2, T1, T2, sample["matches_good_unique_nums"], sample["t_scene"])

    # Visualize
    visalizer = Visualizer()
    visalizer.show(sample, weights, F_est, T1, T2, out_layers)
    

def main():
    # model files check and download
    check_and_download_models(WEIGHT_INPUT_WEIGHTS_PATH, MODEL_INPUT_WEIGHTS_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_UPDATE_WEIGHTS_PATH, MODEL_UPDATE_WEIGHTS_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net_1 = ailia.Net(MODEL_INPUT_WEIGHTS_PATH, WEIGHT_INPUT_WEIGHTS_PATH, env_id=env_id)
    net_2 = ailia.Net(MODEL_UPDATE_WEIGHTS_PATH, WEIGHT_UPDATE_WEIGHTS_PATH, env_id=env_id)
    net = DeepFNet(net_1, net_2)

    recognize_from_image(net)


if __name__ == '__main__':
    main()