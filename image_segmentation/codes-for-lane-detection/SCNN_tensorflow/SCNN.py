import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = '00000.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 288    # net.get_input_shape()[3]
IMAGE_WIDTH = 800     # net.get_input_shape()[2]
OUTPUT_HEIGHT = 288  # net.get_output_shape()[3]
OUTPUT_WIDTH = 800   # net.get_output.shape()[2]


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Lane Detection Model (SCNN Tensorflow)', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-n', '--normal', action='store_true',
    help=('By default, the optimized model is used, but with this option, ' +
          'you can switch to the normal (not optimized) model')
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
if not args.normal:
    WEIGHT_PATH = 'SCNN_tensorflow.opt.onnx'
    MODEL_PATH = 'SCNN_tensorflow.opt.onnx.prototxt'
else:
    WEIGHT_PATH = 'SCNN_tensorflow.onnx'
    MODEL_PATH = 'SCNN_tensorflow.onnx.prototxt'
REMOTE_PATH = ''

# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    #print("net.get_input_shape: ", net.get_input_shape())
    #print("net.get_output_shape: ", net.get_output_shape())

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        image = cv2.imread(image_path, int(True))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
        #image = image.transpose((2, 0, 1))  # channel first
        x = image[np.newaxis, :, :, :] # (batch_size, channel, h, w)
        input_data = x.astype(np.float32)
        #print(input_data)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            output, output_exist = net.predict([input_data])

        # postprocessing

        #print("preds_ailia length: ", len(preds_ailia))
        #print("preds_ailia[1]: ", preds_ailia[1])
        #print("preds_ailia[0] shape: ", preds_ailia[0].shape)

        init = False
        for cnt in range(4):
            prob_map = (output[0][:, :, cnt + 1] * 255).astype(int)
            if not init:
                if output_exist[0][cnt] > 0.5:
                    out_img = prob_map
                    init = True
            else:
                if output_exist[0][cnt] > 0.5:
                    out_img += prob_map

        if not init:
            logger.info('Script finished successfully, but no output image.')
            return

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out_img)

    logger.info('Script finished successfully.')

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    recognize_from_image()


if __name__ == '__main__':
    main()