import os
import sys
import time

import numpy as np
import cv2

import ailia
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
    SEGMENTATION_WEIGHT_PATH = 'u2net_opset11.onnx'
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


def safe_crop(mat, crop_pos=(0,0), crop_size=(img_rows, img_cols)):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    x = crop_pos[0]
    y = crop_pos[1]
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


def dump_segmentation(pred, src_data, w, h):
    savedir = os.path.dirname(args.savepath)
    segmentation_data = cv2.resize(pred * 255, (w, h))
    segmentation_data = cv2.cvtColor(segmentation_data, cv2.COLOR_GRAY2BGR)
    segmentation_data = (src_data + segmentation_data)/2
    cv2.imwrite(os.path.join(savedir, "debug_segmentation.png"), segmentation_data)


def dump_segmentation_threshold(trimap_data, src_data, w, h):
    savedir = os.path.dirname(args.savepath)
    segmentation_data = trimap_data.copy()
    segmentation_data = cv2.cvtColor(segmentation_data, cv2.COLOR_GRAY2BGR)
    segmentation_data = segmentation_data.astype(np.float)
    segmentation_data = (src_data + segmentation_data)/2
    cv2.imwrite(
        os.path.join(savedir, "debug_segmentation_threshold.png"),
        segmentation_data,
    )


def dump_trimap(trimap_data, src_data, w, h):
    savedir = os.path.dirname(args.savepath)

    cv2.imwrite(
        os.path.join(savedir, "debug_trimap_gray.png"),
        trimap_data,
    )

    segmentation_data = trimap_data.copy().astype(np.uint8)
    segmentation_data = cv2.cvtColor(segmentation_data, cv2.COLOR_GRAY2BGR)
    segmentation_data = segmentation_data.astype(np.float)
    segmentation_data = (src_data + segmentation_data)/2

    cv2.imwrite(
        os.path.join(savedir, "debug_trimap.png"),
        segmentation_data,
    )


def generate_trimap(net, input_data):
    src_data = input_data.copy()
    
    h = input_data.shape[0]
    w = input_data.shape[1]

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
        dump_segmentation(pred, src_data, w, h)

    if args.arch == "u2net":
        pred = norm(pred)
    if args.arch == "pspnet":
        pred = sigmoid(pred)

    trimap_data = cv2.resize(pred * 255, (w, h))
    trimap_data = trimap_data.reshape((h, w, 1))

    seg_data = trimap_data.copy()

    thre = 0.6

    thre = 255 * thre
    trimap_data[trimap_data < thre] = 0
    trimap_data[trimap_data >= thre] = 255

    if args.arch == "deeplabv3":
        seg_data = trimap_data.copy()

    trimap_data = trimap_data.astype("uint8")

    if args.debug:
        dump_segmentation_threshold(trimap_data, src_data, w, h)

    trimap_data = erode_and_dilate(trimap_data, k_size=(7, 7), ite=3)

    if args.debug:
        dump_trimap(trimap_data, src_data, w, h)

    return trimap_data, seg_data


def matting_preprocess(src_img, trimap_data, seg_data):
    input_data = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    input_data[:, :, :, 0:3] = src_img[:, :, 0:3]
    input_data[:, :, :, 3] = trimap_data[:, :, 0]

    savedir = os.path.dirname(args.savepath)
    if args.debug:
        cv2.imwrite(
            os.path.join(savedir, "debug_input.png"),
            input_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4)),
        )

    input_data = input_data / 255.0

    #if args.debug:
    #    cv2.imwrite(os.path.join(savedir, "debug_rgb.png"), src_img)
    #    cv2.imwrite(os.path.join(savedir, "debug_trimap.png"), trimap_data)

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
    # net initialize
    env_id = 0  # use cpu because overflow fp16 range
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 4))

    seg_net = ailia.Net(
        SEGMENTATION_MODEL_PATH,
        SEGMENTATION_WEIGHT_PATH,
        env_id=args.env_id,
    )
    #seg_net.set_input_shape((1,3,640,640))

    # color space
    rgb_mode = False

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        src_img = cv2.imread(image_path)    
        if rgb_mode:
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

        # create trimap
        if args.trimap == "":
            input_data = src_img
            trimap_data, seg_data = generate_trimap(seg_net, input_data)
        else:
            trimap_data = cv2.imread(args.trimap)

        # output image buffer
        output_img = np.zeros((src_img.shape[0],src_img.shape[1],4))

        # tile loop
        for y in range((src_img.shape[0]+IMAGE_HEIGHT-1)//IMAGE_HEIGHT):
            for x in range((src_img.shape[1]+IMAGE_WIDTH-1)//IMAGE_WIDTH):
                logger.info('Tile ('+str(x)+','+str(y)+')')

                # crop
                crop_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
                crop_pos = (x * IMAGE_WIDTH, y * IMAGE_HEIGHT)

                src_crop_img = safe_crop(src_img, crop_pos, crop_size)
                trimap_crop_data = safe_crop(trimap_data, crop_pos, crop_size)

                seg_data = trimap_crop_data.copy()

                input_data, src_crop_img, trimap_crop_data = matting_preprocess(
                    src_crop_img, trimap_crop_data, seg_data
                )

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
                    preds_ailia = net.predict(input_data)

                # post-processing
                res_img = postprocess(src_crop_img, trimap_crop_data, preds_ailia)

                # copy
                ch = res_img.shape[0]
                cw = res_img.shape[1]
                if crop_pos[0] + cw >= output_img.shape[1]:
                    cw = output_img.shape[1] - crop_pos[0]
                if crop_pos[1] + ch >= output_img.shape[0]:
                    ch = output_img.shape[0] - crop_pos[1]
                output_img[crop_pos[1]:crop_pos[1]+ch,crop_pos[0]:crop_pos[0]+cw,:] = res_img[0:ch,0:cw,:]
        
        # save
        if rgb_mode:
            output_img = output_img.astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGBA2BGRA)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = 0  # use cpu because overflow fp16 range
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 4))

    seg_net = ailia.Net(
        SEGMENTATION_MODEL_PATH,
        SEGMENTATION_WEIGHT_PATH,
        env_id=args.env_id,
    )

    capture = webcamera_utils.get_capture(args.video)
    writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # grab src image
        src_img = frame

        # limit resolution
        w_limit = 640
        if src_img.shape[0]>=w_limit:
            src_img = cv2.resize(src_img,(w_limit,int(w_limit*src_img.shape[0]/src_img.shape[1])))

        # output image buffer
        output_img = np.zeros((src_img.shape[0],src_img.shape[1],4))

        # create video writer if savepath is specified as video format
        if args.savepath != SAVE_IMAGE_PATH:
            if writer==None:
                writer = webcamera_utils.get_writer(
                    args.savepath, output_img.shape[0], output_img.shape[1]
                )

        # generate trimap
        trimap_data, seg_data = generate_trimap(seg_net, src_img)
        cv2.imshow('trimap', trimap_data / 255.0)
        cv2.imshow('segmentation', seg_data / 255.0)

        # tile loop
        for y in range((src_img.shape[0]+IMAGE_HEIGHT-1)//IMAGE_HEIGHT):
            for x in range((src_img.shape[1]+IMAGE_WIDTH-1)//IMAGE_WIDTH):
                logger.info('Tile ('+str(x)+','+str(y)+')')

                crop_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
                crop_pos = (x * IMAGE_WIDTH, y * IMAGE_HEIGHT)

                src_img_crop = safe_crop(src_img, crop_pos, crop_size)
                trimap_data_crop = safe_crop(trimap_data, crop_pos, crop_size)

                input_data, src_img_crop, trimap_data_crop = matting_preprocess(
                    src_img_crop, trimap_data_crop, seg_data
                )

                preds_ailia = net.predict(input_data)

                # postprocessing
                res_img = postprocess(src_img_crop, trimap_data_crop, preds_ailia)
                res_img = composite(res_img)

                # copy
                ch = res_img.shape[0]
                cw = res_img.shape[1]
                if crop_pos[0] + cw >= output_img.shape[1]:
                    cw = output_img.shape[1] - crop_pos[0]
                if crop_pos[1] + ch >= output_img.shape[0]:
                    ch = output_img.shape[0] - crop_pos[1]
                output_img[crop_pos[1]:crop_pos[1]+ch,crop_pos[0]:crop_pos[0]+cw,:] = res_img[0:ch,0:cw,:]

        cv2.imshow('masked', output_img / 255.0)

        # save results
        if writer is not None:
            writer.write(output_img[:,:,0:3].astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(
        SEGMENTATION_WEIGHT_PATH,
        SEGMENTATION_MODEL_PATH,
        SEGMENTATION_REMOTE_PATH,
    )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
