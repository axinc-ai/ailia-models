import time
import sys
import numpy as np
import torch

import ailia

# import original moduls
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
#WEIGHT_PATH = "wavenet_pytorch.onnx"
#MODEL_PATH = "wavenet_pytorch.onnx.prototxt"
#REMOTE_PATH = "../../../test/"
WEIGHT_PATH = "./wavenet_pytorch.onnx"
MODEL_PATH = "./wavenet_pytorch.onnx.prototxt"


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('pytorch_wavenet', './dammy.wav', './output.txt', input_ftype='audio')
args = update_parser(parser)


# ======================
# Main function
# ======================
def main():
    # use start data
    data = np.random.rand(16, 256, 1028)

    # model files check and download
    #check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # create instance
    #env_id = ailia.get_gpu_environment_id()
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    # inference
    for i in range(10):
        tic = time.time()
        preds_ailia = net.predict(data) # (128,256)
        toc = time.time()
        print("ailia processing does take {} seconds".format(str((toc-tic)*0.01)))

    logger.info('Script finished successfully.')

if __name__ == "__main__":
    main()
