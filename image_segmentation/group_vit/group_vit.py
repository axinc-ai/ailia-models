import sys
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt

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

WEIGHT_PATH = 'group_vit_gcc_yfcc_30e-74d335e6.onnx'
MODEL_PATH = 'group_vit_gcc_yfcc_30e-74d335e6.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/group_vit/'

CLASSES = (
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'table', 'dog', 'horse',
    'motorbike', 'person', 'plant', 'sheep', 'sofa', 'train', 'monitor'
)

IMAGE_PATH = 'voc.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'GroupViT', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def seg2coord(seg_map):
    """
    Args:
        seg_map (np.ndarray): (H, W)

    Return:
        dict(group_id -> (x, y))
    """
    h, w = seg_map.shape
    # [h ,w, 2]
    coords = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), axis=-1)
    labels = np.unique(seg_map)
    coord_map = {}
    for label in labels:
        coord_map[label] = coords[seg_map == label].mean(axis=0)
    return coord_map


def blend_result(img, seg, opacity=0.5):
    palette = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
        [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
    ])

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    fg_mask = seg != 0
    img[fg_mask] = img[fg_mask] * (1 - opacity) + color_seg[fg_mask] * opacity

    img = img.astype(np.uint8)

    return img


def show_result(img, pred):
    labels = np.unique(pred)
    coord_map = seg2coord(pred)

    blended_img = blend_result(img, pred, opacity=0.5)
    blended_img = blended_img[:, :, ::-1]  # BGR -> RGB

    width, height = img.shape[1], img.shape[0]
    EPS = 1e-2
    fig = plt.figure(frameon=False)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    for i, label in enumerate(labels):
        if label == 0:
            continue
        center = coord_map[label].astype(np.int32)
        label_text = CLASSES[label]
        ax.text(
            center[1],
            center[0],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.5,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color='orangered',
            fontsize=16,
            verticalalignment='top',
            horizontalalignment='left')

    plt.imshow(blended_img)
    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    plt.close()

    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = img[:, :, ::-1]

    return img


# ======================
# Main functions
# ======================


def preprocess(img):
    img = normalize_image(img, normalize_type='ImageNet')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def predict(net, img):
    h, w, _ = img.shape

    img = img[:, :, ::-1]  # BGR -> RGB
    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    seg_logit = output[0]

    seg_pred = np.argmax(seg_logit, axis=1)

    return seg_pred[0]


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

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net, img)

        res_img = show_result(img, pred)

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
        out = predict(net, img)
        pred = out[2]

        # plot result
        res_img = show_result(frame, pred)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            res_img = res_img.astype(np.uint8)
            writer.write(res_img)

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
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
