import sys
import time

import numpy as np
import cv2

import ailia
# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'deep-image-matting.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deep-image-matting/'

IMAGE_PATH = 'input.png'
TRIMAP_PATH = 'trimap.png'

SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320

MODEL_LISTS = ['deeplabv3', 'u2net', 'pspnet']

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Deep Image Matting', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-t', '--trimap', metavar='IMAGE',
    default=TRIMAP_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='deeplabv3', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-d', '--debug',
    action='store_true',
    help='Dump debug images'
)
args = update_parser(parser)

if args.arch == 'u2net':
    SEGMENTATION_WEIGHT_PATH = 'u2net.onnx'
    SEGMENTATION_MODEL_PATH = SEGMENTATION_WEIGHT_PATH + '.prototxt'
    SEGMENTATION_REMOTE_PATH = \
        'https://storage.googleapis.com/ailia-models/u2net/'
elif args.arch == 'deeplabv3':
    SEGMENTATION_WEIGHT_PATH = 'deeplabv3.opt.onnx'
    SEGMENTATION_MODEL_PATH = SEGMENTATION_WEIGHT_PATH + '.prototxt'
    SEGMENTATION_REMOTE_PATH = \
        'https://storage.googleapis.com/ailia-models/deeplabv3/'
elif args.arch == 'pspnet':
    SEGMENTATION_WEIGHT_PATH = 'pspnet-hair-segmentation.onnx'
    SEGMENTATION_MODEL_PATH = SEGMENTATION_WEIGHT_PATH + '.prototxt'
    SEGMENTATION_REMOTE_PATH = \
        'https://storage.googleapis.com/ailia-models/pspnet-hair-segmentation/'


# ======================
# Utils
# ======================
img_rows = IMAGE_HEIGHT
img_cols = IMAGE_WIDTH


def safe_crop(mat, crop_size=(img_rows, img_cols)):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    x = 0
    y = 0
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != (img_rows, img_cols):
        ret = cv2.resize(
            ret, dsize=(img_rows, img_cols), interpolation=cv2.INTER_NEAREST
        )
    return ret


def get_final_output(out, trimap):
    unknown_code = 128
    mask = np.equal(trimap, unknown_code).astype(np.float32)
    return (1 - mask) * trimap + mask * out


def postprocess(src_img, trimap, preds_ailia):
    trimap = trimap[:, :, 0].reshape((IMAGE_HEIGHT, IMAGE_WIDTH))

    preds_ailia = preds_ailia.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    preds_ailia = preds_ailia * 255.0
    preds_ailia = get_final_output(preds_ailia, trimap)

    output_data = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    output_data[:, :, 0] = src_img[:, :, 0]
    output_data[:, :, 1] = src_img[:, :, 1]
    output_data[:, :, 2] = src_img[:, :, 2]
    output_data[:, :, 3] = preds_ailia

    output_data[output_data > 255] = 255
    output_data[output_data < 0] = 0

    return output_data


# ======================
# Segmentation util
# ======================
def norm(pred):
    ma = np.max(pred)
    mi = np.min(pred)
    return (pred - mi) / (ma - mi)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def erode_and_dilate(mask, k_size, ite):
    kernel = np.ones(k_size, np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=ite)
    dilated = cv2.dilate(mask, kernel, iterations=ite)
    trimap = np.full(mask.shape, 128)
    trimap[eroded >= 254] = 255
    trimap[dilated <= 1] = 0
    return trimap


def deeplabv3_preprocess(input_data):
    input_data = input_data / 127.5 - 1.0
    input_data = input_data.transpose((2, 0, 1))[np.newaxis, :, :, :]
    return input_data


def imagenet_preprocess(input_data):
    input_data = input_data / 255.0
    input_data[:, :, 0] = (input_data[:, :, 0]-0.485)/0.229
    input_data[:, :, 1] = (input_data[:, :, 1]-0.456)/0.224
    input_data[:, :, 2] = (input_data[:, :, 2]-0.406)/0.225
    input_data = input_data.transpose((2, 0, 1))[np.newaxis, :, :, :]
    return input_data


def generate_trimap(net, input_data):
    w = input_data.shape[1]
    h = input_data.shape[0]

    input_shape = net.get_input_shape()
    input_data = cv2.resize(input_data, (input_shape[2], input_shape[3]))

    if args.arch == "deeplabv3":
        input_data = deeplabv3_preprocess(input_data)
    if args.arch == "u2net" or args.arch == "pspnet":
        input_data = imagenet_preprocess(input_data)

    preds_ailia = net.predict([input_data])

    if args.arch == "deeplabv3":
        pred = preds_ailia[0]
        pred = pred[0, 15, :, :] / 21.0
    if args.arch == "u2net" or args.arch == "pspnet":
        pred = preds_ailia[0][0, 0, :, :]

    if args.debug:
        cv2.imwrite("debug_segmentation.png", pred*255)

    if args.arch == "u2net":
        pred = norm(pred)
    if args.arch == "pspnet":
        pred = sigmoid(pred)

    trimap_data = cv2.resize(pred * 255, (w, h))
    trimap_data = trimap_data.reshape((w, h, 1))

    seg_data = trimap_data.copy()

    thre = 255 * 0.6
    trimap_data[trimap_data < thre] = 0
    trimap_data[trimap_data >= thre] = 255

    if args.arch == "deeplabv3":
        seg_data = trimap_data.copy()

    trimap_data = trimap_data.astype("uint8")

    if args.debug:
        cv2.imwrite("debug_segmentation_threshold.png", trimap_data)

    trimap_data = erode_and_dilate(trimap_data, k_size=(7, 7), ite=3)

    if args.debug:
        cv2.imwrite("debug_trimap_full.png", trimap_data)

    return trimap_data, seg_data


def matting_preprocess(src_img, trimap_data, seg_data):
    input_data = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    input_data[:, :, :, 0:3] = src_img[:, :, 0:3]

    input_data[:, :, :, 3] = trimap_data[:, :, 0]
    if args.debug:
        cv2.imwrite("debug_input.png", input_data.reshape(
            (IMAGE_HEIGHT, IMAGE_WIDTH, 4)))

    input_data = input_data / 255.0

    if args.debug:
        cv2.imwrite("debug_rgb.png", src_img)
        cv2.imwrite("debug_trimap.png", trimap_data)

    return input_data, src_img, trimap_data


def composite(img):
    img[:, :, 0] = img[:, :, 0] * img[:, :, 3] / 255.0
    img[:, :, 1] = img[:, :, 1] * img[:, :, 3] / \
        255.0 + 255.0 * (1.0 - img[:, :, 3] / 255.0)
    img[:, :, 2] = img[:, :, 2] * img[:, :, 3] / 255.0
    return img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    src_img = cv2.imread(args.input)

    crop_size = (
        max(src_img.shape[0], src_img.shape[1]),
        max(src_img.shape[0], src_img.shape[1])
    )

    src_img = safe_crop(src_img, crop_size)

    # net initialize
    env_id = 0  # use cpu because overflow fp16 range
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 4))

    if args.trimap == "":
        input_data = src_img

        seg_net = ailia.Net(
            SEGMENTATION_MODEL_PATH,
            SEGMENTATION_WEIGHT_PATH,
            env_id=args.env_id
        )

        trimap_data, seg_data = generate_trimap(seg_net, input_data)
    else:
        trimap_data = cv2.imread(args.trimap)
        trimap_data = safe_crop(trimap_data, crop_size)
        seg_data = trimap_data.copy()

    input_data, src_img, trimap_data = matting_preprocess(
        src_img, trimap_data, seg_data
    )

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(input_data)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = net.predict(input_data)

    # post-processing
    res_img = postprocess(src_img, trimap_data, preds_ailia)
    cv2.imwrite(args.savepath, res_img)

    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = 0  # use cpu because overflow fp16 range
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 4))

    seg_net = ailia.Net(
        SEGMENTATION_MODEL_PATH,
        SEGMENTATION_WEIGHT_PATH,
        env_id=args.env_id
    )

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        print(
            '[WARNING] currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # grab src image
        src_img, input_data = webcamera_utils.preprocess_frame(
            frame,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            normalize_type='None'
        )
        crop_size = (
            max(src_img.shape[0], src_img.shape[1]),
            max(src_img.shape[0], src_img.shape[1])
        )
        src_img = safe_crop(src_img, crop_size)

        trimap_data, seg_data = generate_trimap(seg_net, src_img)

        input_data, src_img, trimap_data = matting_preprocess(
            src_img, trimap_data, seg_data
        )

        preds_ailia = net.predict(input_data)

        # postprocessing
        res_img = postprocess(src_img, trimap_data, preds_ailia)
        seg_img = res_img.copy()

        # seg_data=safe_crop(seg_data, x, y, crop_size)
        seg_img[:, :, 3] = seg_data[:, :, 0]
        seg_img = composite(seg_img)
        res_img = composite(res_img)

        cv2.imshow('matting', res_img / 255.0)
        cv2.imshow('masked', seg_img / 255.0)
        cv2.imshow('trimap', trimap_data / 255.0)
        cv2.imshow('segmentation', seg_data / 255.0)

        # # save results
        # if writer is not None:
        #     writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(
        SEGMENTATION_WEIGHT_PATH,
        SEGMENTATION_MODEL_PATH,
        SEGMENTATION_REMOTE_PATH
    )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
