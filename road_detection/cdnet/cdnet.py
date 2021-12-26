import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'cdnet.onnx'
MODEL_PATH = 'cdnet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/cdnet/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 640

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'CDNet', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    xy = np.stack((xv, yv), axis=2).reshape((1, 1, ny, nx, 2))
    xy = xy.astype(np.float32)

    return xy


# def draw_bbox(img, bboxes):
#     return img


# ======================
# Main functions
# ======================

cache = {
    "grid": [None, None, None],
    "anchor_grid": np.load("anchor_grid.npy")
}


def preprocess(img):
    h, w = (IMAGE_SIZE, IMAGE_SIZE)
    im_h, im_w, _ = img.shape

    # resize
    r = min((h / im_h), (w / im_w))
    ow, oh = int(im_w * r), int(im_h * r)
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    control_line = (400, 680)
    x1, x2 = 0, ow
    y1, y2 = control_line[0] / im_h * oh, control_line[1] / im_h * oh
    img = img[int(y1):int(y2), :]

    rest_h = img.shape[0] % 32
    rest_w = img.shape[1] % 32
    dh = 0 if rest_h == 0 else (32 - rest_h) / 2
    dw = 0 if rest_w == 0 else (32 - rest_w) / 2
    recover_xy = [ow, oh, int(x1) - dw, int(y1) - dh]

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # add border
    color = (114, 114, 114)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    img = img[:, :, ::-1]  # BGR -> RGB
    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, recover_xy


def post_processing(x):
    stride = [8, 16, 32]
    grid = cache['grid']
    anchor_grid = cache['anchor_grid']

    z = []
    for i, _ in enumerate(x):
        bs, _, ny, nx, _ = x[i].shape
        if grid[i] is None or grid[i].shape[2:4] != x[i].shape[2:4]:
            grid[i] = make_grid(nx, ny)

        y = sigmoid(x[i])
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.reshape((bs, -1, 7)))

    z = np.concatenate(z, axis=1)

    return z


def predict(net, img):
    img, _ = preprocess(img)

    # feedforward
    output = net.predict([img])

    pred = post_processing(output)

    return pred


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                pred = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net, img)

        res_img = img

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = predict(net, img)

        res_img = img

        # show
        cv2.imshow('frame', res_img)

        # save results
        if writer is not None:
            writer.write(res_img.astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
