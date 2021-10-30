import sys
import os
import time
import glob

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

from utils_sh import get_shading

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'trained_model_03.onnx'
MODEL_PATH = 'trained_model_03.onnx.prototxt'
WEIGHT_1024_PATH = 'trained_model_1024_03.onnx'
MODEL_1024_PATH = 'trained_model_1024_03.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/dpr/'

IMAGE_PATH = 'obama.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 512
IMAGE_1024_SIZE = 1024

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'DPR', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-sh', metavar='FILE/DIR', default='lighting/rotate_light_00.txt',
    help='SH (Spherical Harmonics) lighting data'
)
parser.add_argument(
    '--shading', action='store_true',
    help='rendering half-sphere'
)
parser.add_argument(
    '-m', '--model_type', default='512', choices=('512', '1024'),
    help='model type'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def read_sh():
    sh_data = {}
    if os.path.isdir(args.sh):
        for path in glob.glob(os.path.join(args.sh, '*.txt')):
            sh = np.loadtxt(path)
            key = os.path.splitext(os.path.basename(path))[0]
            sh_data[key] = sh[:9] * 0.7
    else:
        sh = np.loadtxt(args.sh)
        key = os.path.splitext(os.path.basename(args.sh))[0]
        sh_data[key] = sh[:9] * 0.7

    assert 0 < len(sh_data), 'SH file not found'

    return sh_data


def render_shading(sh_data):
    # create normal for rendering half sphere
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x ** 2 + z ** 2)
    valid = mag <= 1
    y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    logger.info('Save shading...')
    for key, sh in sorted(sh_data.items()):
        sh = np.squeeze(sh)
        shading = get_shading(normal, sh)
        value = np.percentile(shading, 95)
        ind = shading > value
        shading[ind] = value
        shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
        shading = (shading * 255.0).astype(np.uint8)
        shading = np.reshape(shading, (256, 256))
        shading = shading * valid

        savepath = '%s.png' % key
        savepath = get_savepath(
            args.savepath if os.path.isdir(args.savepath) else savepath,
            savepath, post_fix='')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, shading)


# ======================
# Main functions
# ======================

def preprocess(img, image_shape):
    h, w = image_shape

    img = cv2.resize(img, (h, w))
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    img_l = img_lab[:, :, 0]
    img_l = img_l.astype(np.float32) / 255.0
    img_l = img_l.transpose((0, 1))
    img_l = img_l[None, None, ...]

    return img_lab, img_l


def post_processing(img_lab, out_l, image_shape):
    h, w = image_shape

    out_l = out_l.transpose((1, 2, 0))
    out_l = np.squeeze(out_l)
    out_l = (out_l * 255.0).astype(np.uint8)
    img_lab[:, :, 0] = out_l

    out_img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    out_img = cv2.resize(out_img, (w, h))

    return out_img


def predict(net, img, sh_data):
    model_type = args.model_type

    h, w = img.shape[:2]

    img_size = IMAGE_SIZE if model_type == '512' else IMAGE_1024_SIZE
    img_lab, img_l = preprocess(img, (img_size, img_size))

    out_imgs = []
    for key, sh in sorted(sh_data.items()):
        sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)

        if not args.video:
            logger.info(f'SH : {key}')

        # feedforward
        output = net.predict([img_l, sh])
        if model_type == '512':
            out_l, out_light = output
        else:
            out_l, _, out_light = output

        img = post_processing(img_lab, out_l[0], (h, w))
        out_imgs.append(img)

    return out_imgs


def recognize_from_image(net):
    sh_data = read_sh()

    if args.shading:
        render_shading(sh_data)

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
                out_imgs = predict(net, img, sh_data)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out_imgs = predict(net, img, sh_data)

        # save result
        if len(out_imgs) == 1:
            savepath = get_savepath(args.savepath, image_path, ext='.png')
            logger.info(f'saved at : {savepath}')
            cv2.imwrite(savepath, out_imgs[0])
        else:
            for i, name in enumerate(sorted(sh_data.keys())):
                savepath = '%s_%s.png' % (os.path.splitext(os.path.basename(image_path))[0], name)
                savepath = get_savepath(
                    args.savepath if os.path.isdir(args.savepath) else savepath,
                    savepath, post_fix='')
                logger.info(f'saved at : {savepath}')
                cv2.imwrite(savepath, out_imgs[i])

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

    sh_data = read_sh()
    if 1 < len(sh_data):
        # Squeeze to only one
        key = list(sh_data.keys())[0]
        sh_data = {key: sh_data[key]}

    if args.shading:
        render_shading(sh_data)

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # inference
        out_imgs = predict(net, frame, sh_data)

        # show
        cv2.imshow('frame', out_imgs[0])

        # save results
        if writer is not None:
            writer.write(out_imgs[0].astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        '512': (WEIGHT_PATH, MODEL_PATH),
        '1024': (WEIGHT_1024_PATH, MODEL_1024_PATH),
    }
    weight_path, model_path = dic_model[args.model_type]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(model_path, weight_path, env_id=env_id)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
