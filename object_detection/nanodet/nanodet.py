import numpy as np
import time
import os
import sys
import cv2
import matplotlib.pyplot as plt

from nanodet_utils import NanoDetABC

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

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/nanodet/'
WEIGHT_PATH = "nanodet-EfficientNet-Lite0_320.opt.onnx"
MODEL_PATH = "nanodet-EfficientNet-Lite0_320.opt.onnx.prototxt"

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

HEIGHT = 320
WIDTH = 320

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('nanodet model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)


# ======================
# Detection function
# ======================
class NanoDetDetection(NanoDetABC):
    def __init__(self, model, *args, **kwargs):
        super(NanoDetDetection, self).__init__(*args, **kwargs)
        self.model = model

    def infer_image(self, img_input):
        inference_results = self.model.run(img_input)
        scores = [np.squeeze(x) for x in inference_results[:3]]
        raw_boxes = [np.squeeze(x) for x in inference_results[3:]]
        return scores, raw_boxes


# ======================
# Main functions
# ======================
def recognize_from_image():
    env_id = ailia.get_gpu_environment_id()
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    detector = NanoDetDetection(net)

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
                bbox, label, score = detector.detect(raw_img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            bbox, label, score = detector.detect(raw_img)
            img_draw = detector.draw_box(raw_img, bbox, label, score)
            plt.imshow(img_draw[..., ::-1])
            plt.axis('off')
            plt.show()

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img_draw)

    if cv2.waitKey(0) != 32:  # space bar
        exit()


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    detector = NanoDetDetection(net)

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
        bbox, label, score = detector.detect(raw_img)
        img_draw = detector.draw_box(raw_img, bbox, label, score)
        plt.imshow(img_draw[..., ::-1])

        plt.pause(.01)
        if not plt.get_fignums():
            break

        # save results
        if writer is not None:
            writer.write(img_draw)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


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
