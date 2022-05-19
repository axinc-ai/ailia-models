import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
# from image_utils import load_image, normalize_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

import face_detect_crop
from face_detect_crop import crop_face

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_G_PATH = 'G_unet_2blocks.onnx'
MODEL_G_PATH = 'G_unet_2blocks.onnx.prototxt'
WEIGHT_ARCFACE_PATH = 'scrfd_10g_bnkps.onnx'
MODEL_ARCFACE_PATH = 'scrfd_10g_bnkps.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/sber-swap/'

IMAGE_PATH = 'beckham.jpg'
SOURCE_PATH = 'elon_musk.jpg'
SAVE_IMAGE_PATH = 'output.png'

CROP_SIZE = 224

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

THRESHOLD = 0.4
IOU = 0.45

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'SberSwap', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-src', '--source', default=SOURCE_PATH,
    help='source image'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


# def draw_bbox(img, bboxes):
#     return img


# ======================
# Main functions
# ======================

def preprocess(img, image_shape):
    h, w = image_shape
    im_h, im_w, _ = img.shape

    # adaptive_resize
    scale = h / min(im_h, im_w)
    ow, oh = int(im_w * scale), int(im_h * scale)
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    img = np.array(Image.fromarray(img).resize((ow, oh), Image.BILINEAR))

    # center_crop
    if ow > w:
        x = (ow - w) // 2
        img = img[:, x:x + w, :]
    if oh > h:
        y = (oh - h) // 2
        img = img[y:y + h, :, :]

    # img = normalize_image(img, normalize_type='ImageNet')

    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(output):
    return None


def predict(net, img):
    shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    img = preprocess(img, shape)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'src': img})

    pred = post_processing(output)

    return pred


def recognize_from_image(net_iface, net_G):
    source_path = args.source
    logger.info('SOURCE: {}'.format(source_path))

    src_img = load_image(source_path)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGRA2BGR)

    src_img = crop_face(src_img, net_iface, CROP_SIZE)
    src_img = src_img[:, :, ::-1]  # BRG -> RGB

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
                out = predict(net_G, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out = predict(net_G, img)

        # res_img = draw_bbox(out)
        #
        # # plot result
        # savepath = get_savepath(args.savepath, image_path, ext='.png')
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net_iface, net_G):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = predict(net_G, img)

        # # plot result
        # res_img = draw_bbox(frame, out)
        #
        # # show
        # cv2.imshow('frame', res_img)
        # frame_shown = True
        #
        # # save results
        # if writer is not None:
        #     res_img = res_img.astype(np.uint8)
        #     writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('Checking G model...')
    check_and_download_models(WEIGHT_G_PATH, MODEL_G_PATH, REMOTE_PATH)
    logger.info('Checking arcface model...')
    check_and_download_models(WEIGHT_ARCFACE_PATH, MODEL_ARCFACE_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net_iface = ailia.Net(MODEL_ARCFACE_PATH, WEIGHT_ARCFACE_PATH, env_id=env_id)
        net_G = ailia.Net(MODEL_G_PATH, WEIGHT_G_PATH, env_id=env_id)
    else:
        import onnxruntime
        net_iface = onnxruntime.InferenceSession(WEIGHT_ARCFACE_PATH)
        net_G = onnxruntime.InferenceSession(WEIGHT_G_PATH)
        face_detect_crop.onnx = True

    if args.video is not None:
        recognize_from_video(net_iface, net_G)
    else:
        recognize_from_image(net_iface, net_G)


if __name__ == '__main__':
    main()
