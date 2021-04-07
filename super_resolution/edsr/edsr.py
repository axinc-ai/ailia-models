import cv2
import os
import sys
import ailia
import numpy as np
import time

# import original modules
sys.path.append('../../util')
from image_utils import load_image, get_image_shape  # noqa: E402
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 
# ======================

WEIGHT_PATH = 'edsr.onnx'
MODEL_PATH = 'edsr.onnx.prototxt'
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('EDSR model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def recognize_from_image():
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=-1)
    logger.info(IMAGE_PATH)
    for image_path in args.input:

        IMAGE_HEIGHT, IMAGE_WIDTH = get_image_shape(image_path)
        # prepare input data
        logger.info(image_path)
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            gen_input_ailia=True,
            normalize_type='None'
        )
        net.set_input_shape((1,3,IMAGE_HEIGHT,IMAGE_WIDTH))
        #input_data = np.load('np_save.npy')

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
            preds_ailia = net.predict(input=input_data)[0]

        # postprocessing
        output_img = preds_ailia.transpose(1, 2, 0)
        output_img = np.clip(output_img, 0, 255)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=-1)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    time.sleep(1)  

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        IMAGE_HEIGHT, IMAGE_WIDTH = frame.shape[0], frame.shape[1]
        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='None'
        )
        net.set_input_shape((1,3,IMAGE_HEIGHT,IMAGE_WIDTH))

        # Inference
        preds_ailia = net.predict(input_data)[0] 
        output_img = preds_ailia.transpose(1, 2, 0) / 255
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', output_img)
        output_img *= 255
        output_img = np.clip(output_img, 0, 255)     
        #cv2.imwrite('pred.png', output_img) #please uncomment to save output

        # save results
        if writer is not None:
            writer.write(output_img)

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
