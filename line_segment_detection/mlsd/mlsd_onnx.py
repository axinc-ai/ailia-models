import numpy as np
import os
import onnxruntime
import cv2
from PIL import Image

from mlsd_utils import pred_lines_onnx


# logger
from logging import getLogger

logger = getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mlsd/'
WEIGHT_PATH = 'M-LSD_512_large.opt.onnx'
MODEL_PATH = 'M-LSD_512_large.opt.onnx.prototxt'
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# ======================
# Main functions
# ======================
def gradio_wrapper_for_LSD(img_input, sess, input_details, output_details):
  lines = pred_lines_onnx(img_input, sess, input_details, output_details, input_shape=[512, 512])
  img_output = img_input.copy()

  # draw lines
  for line in lines:
    x_start, y_start, x_end, y_end = [int(val) for val in line]
    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [0,255,255], 2)

  return img_output


def recognize_from_image():
    # net initialize
    sess = onnxruntime.InferenceSession(WEIGHT_PATH)
    input_details = sess.get_inputs()
    output_details = sess.get_outputs()

    # input image
    img_input = np.array(Image.open(IMAGE_PATH))
    preds_img = gradio_wrapper_for_LSD(img_input, sess, input_details, output_details)

    # postprocessing
    logger.info(f'saved at : {SAVE_IMAGE_PATH}')
    cv2.imwrite(SAVE_IMAGE_PATH, preds_img)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    recognize_from_image()


if __name__ == '__main__':
    main()
