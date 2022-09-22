from unicodedata import category
import cv2 
import sys
import numpy as np
import time

import ailia

from qrcode_wechatqrcode_utils import preprocess, postprocess, reverse_letterbox

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================
MODEL_PARAMS = {'detect_2021nov': {'input_shape': [384, 384], 'max_stride': 32, 'anchors':[
                    [12,16, 19,36, 40,28], [36,75, 76,55, 72,146], [142,110, 192,243, 459,401]
                    ]},
                }

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov7/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('QR detection model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-m', '--model_name',
    default='detect_2021nov',
)
parser.add_argument(
    '-w', '--write_prediction',
    action='store_true',
    help='Flag to output the prediction file.'
)
args = update_parser(parser)

MODEL_NAME = args.model_name
WEIGHT_PATH = MODEL_NAME + ".caffemodel"
MODEL_PATH = MODEL_NAME + ".prototxt"

HEIGHT = MODEL_PARAMS[MODEL_NAME]['input_shape'][0]
WIDTH = MODEL_PARAMS[MODEL_NAME]['input_shape'][1]
STRIDE = MODEL_PARAMS[MODEL_NAME]['max_stride']
ANCHORS = MODEL_PARAMS[MODEL_NAME]['anchors']

def visualize(image, res, points, points_color=(0, 255, 0), text_color=(0, 255, 0), fps=None):
    output = image.copy()
    h, w, _ = output.shape

    if fps is not None:
        cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    fontScale = 0.5
    fontSize = 1
    for r, p in zip(res, points):
        p = p.astype(np.int32)
        for _p in p:
            cv2.circle(output, _p, 10, points_color, -1)

        qrcode_center_x = int((p[0][0] + p[2][0]) / 2)
        qrcode_center_y = int((p[0][1] + p[2][1]) / 2)

        text_size, baseline = cv2.getTextSize(r, cv2.FONT_HERSHEY_DUPLEX, fontScale, fontSize)
        text_x = qrcode_center_x - int(text_size[0] / 2)
        text_y = qrcode_center_y - int(text_size[1] / 2)
        cv2.putText(output, '{}'.format(r), (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize)

    return output

def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        raw_img = cv2.imread(image_path)
        logger.debug(f'input image shape: {raw_img.shape}')

        img = preprocess(raw_img, (HEIGHT, WIDTH))

        # inference
        logger.info('Start inference...')
        res = None
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                res = net.run(img[None, None, :, :])
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            res = net.run(img[None, None, :, :])

        detections = postprocess(img, res)
        detections = reverse_letterbox(detections, raw_img.shape, img.shape)

        for d in detections:
            cv2.rectangle(raw_img, (int(d.x), int(d.y)), (int(d.x + d.w), int(d.y + d.h)), (255, 0, 0))

        cv2.imshow("QR", raw_img)
        cv2.waitKey()

        savepath = get_savepath(args.savepath, image_path)
        cv2.imwrite(savepath, raw_img)
        logger.info(f'saved at : {savepath}')

    logger.info('Script finished successfully.')

def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while(True):
        ret, raw_frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        frame = preprocess(raw_frame, (HEIGHT, WIDTH))
        
        res = net.run(frame[None, None, :, :])
        detections = postprocess(frame, res)
        detections = reverse_letterbox(detections, raw_frame.shape, frame.shape)

        for d in detections:
            cv2.rectangle(raw_frame, (int(d.x), int(d.y)), (int(d.x + d.w), int(d.y + d.h)), (255, 0, 0))

        cv2.imshow('frame', raw_frame)

        # save results
        if writer is not None:
            writer.write(raw_frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    # check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, 1, WIDTH, HEIGHT))

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)

if __name__ == '__main__':
    main()
