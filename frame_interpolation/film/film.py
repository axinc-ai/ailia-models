import sys
import os
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'film_net.onnx'
MODEL_PATH = 'film_net.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/film/'

IMAGE_PATH = 'photos'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'FILM: Frame Interpolation for Large Motion', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-i2', '--input2', metavar='IMAGE2', default=None,
    help='The second input image path.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    h, w, _ = img.shape
    align = 32

    pad_h = pad_w = 0
    if h % align != 0 or w % align != 0:
        pad_h = align - h % align
        pad_w = align - w % align

        pad_img = np.zeros(h + align - pad_h, w + align - pad_w, 3)
        pad_h = pad_h // 2
        pad_w = pad_w // 2
        pad_img[pad_h:pad_h + h, pad_w:pad_w + w, 3] = img
        img = pad_img

    img = img / 255
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, (pad_h, pad_w)


def predict(net, img1, img2):
    h, w, _ = img1.shape
    img1 = img1[:, :, ::-1]  # BGR -> RGB
    img2 = img2[:, :, ::-1]  # BGR -> RGB

    x0, pad_hw = preprocess(img1)
    x1, _ = preprocess(img2)
    batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

    # feedforward
    if not args.onnx:
        output = net.predict([batch_dt[..., np.newaxis], x0, x1])
    else:
        output = net.run(None, {'time': batch_dt[..., np.newaxis], 'x0': x0, 'x1': x1})

    image = output[24]

    image = np.clip(image[0] * 255, 0, 255)
    image = (image + 0.5).astype(np.uint8)
    image = image[:, :, ::-1]  # RGB -> BGR

    pad_h, pad_w = pad_hw
    if pad_h or pad_w:
        image = image[pad_h:pad_h + h, pad_w:pad_w + w, ...]

    return image


def recognize_from_image(net):
    # Load images
    inputs = args.input
    n_input = len(inputs)
    if n_input == 1 and args.input2:
        inputs.extend([args.input2])

    if len(inputs) < 2:
        logger.error("Specified input must be at least two or more images")
        sys.exit(-1)

    for no, image_paths in enumerate(zip(inputs, inputs[1:])):
        logger.info(image_paths)

        # prepare input data
        images = [load_image(p) for p in image_paths]
        img1, img2 = [cv2.cvtColor(im, cv2.COLOR_BGRA2BGR) for im in images]

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                out_img = predict(net, img1, img2)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out_img = predict(net, img1, img2)

        nm_ext = os.path.splitext(SAVE_IMAGE_PATH)
        save_file = "%s_%s%s" % (nm_ext[0], no, nm_ext[1])
        save_path = get_savepath(args.savepath, save_file, post_fix='', ext='.png')
        logger.info(f'saved at : {save_path}')
        cv2.imwrite(save_path, out_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if 0 < video_length:
        logger.info(f"video_length: {video_length}")

    # create video writer if savepath is specified as video format
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer = None
    if args.savepath != SAVE_IMAGE_PATH:
        writer = get_writer(args.savepath, f_h, f_w, fps=fps)

    # create output buffer
    n_output = 1
    output_buffer = np.zeros((f_h * (n_output + 2), f_w, 3))
    output_buffer = output_buffer.astype(np.uint8)

    images = []

    it = None
    try:
        import tqdm
        if 0 < video_length:
            it = iter(tqdm(range(video_length)))
            next(it)
    except ImportError:
        pass

    frame_shown = False
    while True:
        if it and 0 < video_length:
            try:
                next(it)
            except StopIteration:
                break
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # set inputs
        images.append(cv2.resize(frame, (f_w, f_h)))
        if len(images) < 2:
            continue
        elif len(images) > 2:
            images = images[1:]

        # inference
        img1, img2 = images
        out_img = predict(net, img1, img2)

        output_buffer[:f_h, :f_w, :] = images[0]
        output_buffer[f_h * 1:f_h * 2, :f_w, :] = out_img
        output_buffer[f_h * 2:f_h * 3, :f_w, :] = images[1]

        # preview
        cv2.imshow('frame', output_buffer)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(images[0])
            writer.write(out_img)

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
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
