
import sys
import time
import numpy as np
import skimage
import cv2

import onnxruntime
import ailia

from mrcnn.config import Config
from mrcnn import utils

config = Config()

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters 1
# ======================
IMAGE_PATH = '48001.jpg'
SAVE_IMAGE_PATH = 'output.png'
WEIGHT_PATH = 'mask_rcnn_poles_resnet101.onnx'
MODEL_PATH = 'mask_rcnn_poles_resnet101.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/detect-utility-poles/'

SLEEP_TIME = 0

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'POLE_DETECTION,.', IMAGE_PATH, SAVE_IMAGE_PATH,
)

parser.add_argument(
    '-i', '--input', type=str,
    default=IMAGE_PATH,
    help='The input image for pole detection.'
)
parser.add_argument(
    '-v', '--video', type=str,
    help='The input video for pole detection.'
)
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
args = update_parser(parser)

# # ======================
# # Main functions
# # ======================
def recognize_from_image():
    # net initialize
    if args.onnx:
        import onnxruntime
        model = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        logger.info(f'env_id: {args.env_id}')
        model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
        
    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)
    molded_images = np.expand_dims(molded_image, axis=0)
    molded_images = molded_images.astype(np.float32) - config.MEAN_PIXEL

    image_metas = utils.compose_image_meta(
            0, image.shape, molded_image.shape, window, scale,
            np.zeros([config.NUM_CLASSES], dtype=np.int32))
    image_metas = np.expand_dims(image_metas, axis=0)

    anchors = utils.get_anchors(molded_images[0].shape)
    anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)

    if args.onnx:
        results = \
            model.run(None, {"input_image": molded_images.astype(np.float32),
                            "input_anchors": anchors,
                            "input_image_meta": image_metas.astype(np.float32)})
    else:
        results = \
            model.run({"input_image": molded_images.astype(np.float32),
                            "input_anchors": anchors,
                            "input_image_meta": image_metas.astype(np.float32)})

    images = [image]
    windows = [window]
    results_final, result_image = utils.generate_image(images, molded_images, windows, results)

    cv2.imwrite( SAVE_IMAGE_PATH, result_image)


def recognize_from_video():
    # net initialize
    if args.onnx:
        import onnxruntime
        model = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        logger.info(f'env_id: {args.env_id}')
        model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    while(True):

        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
         
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        molded_image, window, scale, padding, crop = utils.resize_image(
                    image,
                    min_dim=config.IMAGE_MIN_DIM,
                    min_scale=config.IMAGE_MIN_SCALE,
                    max_dim=config.IMAGE_MAX_DIM,
                    mode=config.IMAGE_RESIZE_MODE)
        molded_images = np.expand_dims(molded_image, axis=0)
        molded_images = molded_images.astype(np.float32) - config.MEAN_PIXEL
        
        image_metas = utils.compose_image_meta(
            0, image.shape, molded_image.shape, window, scale,
            np.zeros([config.NUM_CLASSES], dtype=np.int32))
            
        image_metas = np.expand_dims(image_metas, axis=0)
        anchors = utils.get_anchors(molded_images[0].shape)
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
        
        if args.onnx:
            results = \
            model.run(None, {"input_image": molded_images.astype(np.float32),
                            "input_anchors": anchors,
                            "input_image_meta": image_metas.astype(np.float32)})
        else:
            results = \
            model.run({"input_image": molded_images.astype(np.float32),
                            "input_anchors": anchors,
                            "input_image_meta": image_metas.astype(np.float32)})
                        
        images = [image]
        windows = [window]
        results_final, result_image = utils.generate_image(images, molded_images, windows, results)

        cv2.imshow('frame', result_image)
        time.sleep(SLEEP_TIME)

    capture.release()
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')
    pass

def main():

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()

if __name__ == '__main__':
    main()