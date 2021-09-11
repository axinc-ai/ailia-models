import sys
import time
import numpy as np
import cv2
import onnxruntime
from scipy.special import softmax
import scipy.stats
from PIL import Image as pimg
import matplotlib.pyplot as plt

from max_deeplab_utils.postprosess import  shape_pred, visualize

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402 noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = "max_deeplab.onnx"

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'
HEIGHT = 224
WIDTH = 224
N = 30

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('max_deeplab model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # onnxruntime initialize
    session = onnxruntime.InferenceSession(WEIGHT_PATH)
    input_name = session.get_inputs()[0].name
    output_name1 = session.get_outputs()[0].name
    output_name2 = session.get_outputs()[1].name
    output_name3 = session.get_outputs()[2].name

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = cv2.imread(image_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        logger.debug(f'input image shape: {raw_img.shape}')
        resize_img = cv2.resize(raw_img, (WIDTH, HEIGHT))
        img = scipy.stats.zscore(resize_img)
        img = img.transpose(2, 0, 1)
        img = np.array([img], np.float32)


        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                out = session.run([output_name1, output_name2, output_name3], {input_name: img})
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            out = session.run([output_name1, output_name2, output_name3], {input_name: img})

        # postprocessing
        instances, classes, keep_pred_instances = shape_pred(out, N)
        fig = visualize(resize_img, instances, classes, keep_pred_instances)

        # save
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        fig.savefig(savepath)

        if cv2.waitKey(0) != 32:  # space bar
            exit()


def recognize_from_video():
    # onnxruntime initialize
    session = onnxruntime.InferenceSession(WEIGHT_PATH)
    input_name = session.get_inputs()[0].name
    output_name1 = session.get_outputs()[0].name
    output_name2 = session.get_outputs()[1].name
    output_name3 = session.get_outputs()[2].name

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w, rgb=False)
    else:
        writer = None

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        resize_frame = cv2.resize(frame, (WIDTH, HEIGHT))
        input = scipy.stats.zscore(resize_frame)
        input = input.transpose(2, 0, 1)
        input = np.array([input], np.float32)

        # inference
        pred = session.run([output_name1, output_name2, output_name3], {input_name: input})

        # postprocessing
        instances, classes, keep_pred_instances = shape_pred(pred, N)
        _ = visualize(resize_frame, instances, classes, keep_pred_instances)

        plt.pause(.01)
        if not plt.get_fignums():
            break

        # save results
        if writer is not None:
            writer.write(pred)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
