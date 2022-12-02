import sys
import time

import ailia
import cv2
import numpy as np

sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================
MODEL_PATH = 'pytorch-fcn32s.onnx'
WEIGHTL_PATH = 'pytorch-fcn32s.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pytorch-fcn32s/'
IMAGE_PATH = './demo.jpg'
SAVE_IMAGE_PATH = './out.jpg'
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 640

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'PyTorch-FCN model', IMAGE_PATH, SAVE_IMAGE_PATH
)
# parser.add_argument(
#     # '-m', '--model_type', metavar='MODEL_TYPE',
#     # default='camvid', choices=MODEL_LIST,
#     # help='Set model architecture: ' + ' | '.join(MODEL_LIST)
# )
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)

# ======================
# Secondaty Functions
# ======================
def pre_process(img):
    img = cv2.resize(
        img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA
    )
    img = np.array([[img, img, img]])
    img = img.transpose([2,0,1])
    return img

def post_processing():
    pass

def segment_img(img, net_out):
    pass

def predict(img, net):
    img = pre_process(img)
    print(img)
    pass

# ======================
# Main functions
# ======================
def recognize(net) :
    logger.info(IMAGE_PATH)
    img = load_image(IMAGE_PATH)
    logger.debug(f'input image shape: {img.shape}')

    logger.info('Start inference...')
    net_out = predict(img, net)
    segemantation_image = segment_img(img, net_out)
    savepath = get_savepath(args.savepath, IMAGE_PATH)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, segemantation_image)
    logger.info('Script finished successfully.')

def main():

    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHTL_PATH)
    else:
        net = ailia.Net(MODEL_PATH, WEIGHTL_PATH, env_id = args.env_id)
    recognize(net)

if __name__ == '__main__':
    main()
