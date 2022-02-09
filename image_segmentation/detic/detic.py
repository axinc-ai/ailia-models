import sys
import time
import datetime

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

from dataset_utils import get_lvis_meta_v1
from color_utils import random_color, color_brightness

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO.onnx'
MODEL_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/detic/'

IMAGE_PATH = 'desk.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 800
IMAGE_MAX_SIZE = 1333

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Detic', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--seed', type=int, default=int(datetime.datetime.now().strftime('%Y%m%d')),
    help='random seed'
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

def do_paste_mask(masks, boxes, im_h, im_w):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """

    x0_int, y0_int = 0, 0
    x1_int, y1_int = im_w, im_h
    x0, y0, x1, y1 = np.split(boxes, 4, axis=1)  # each is Nx1

    img_y = np.arange(y0_int, y1_int, dtype=np.float32) + 0.5
    img_x = np.arange(x0_int, x1_int, dtype=np.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1

    gx = np.repeat(img_x[:, None, :], img_y.shape[1], axis=1)
    gy = np.repeat(img_y[:, :, None], img_x.shape[1], axis=2)
    grid = np.stack([gx, gy], axis=3)

    import torch
    from torch.nn import functional as F
    img_masks = F.grid_sample(torch.from_numpy(masks), torch.from_numpy(grid), align_corners=False)
    img_masks = img_masks.numpy()

    return img_masks[:, 0]


def paste_masks_in_image(
        masks, boxes, image_shape, threshold: float = 0.5):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.
    """

    if len(masks) == 0:
        return np.zeros((0,) + image_shape, dtype=np.uint8)

    im_h, im_w = image_shape

    img_masks = do_paste_mask(
        masks[:, None, :, :], boxes, im_h, im_w,
    )
    img_masks = img_masks >= threshold

    return img_masks


def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.

    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False

    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]

    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]

    return res, has_holes


def draw_predictions(img, predictions):
    height, width = img.shape[:2]

    boxes = predictions["pred_boxes"].astype(np.int64)
    scores = predictions["scores"]
    classes = predictions["pred_classes"].tolist()
    masks = predictions["pred_masks"].astype(np.uint8)

    class_names = get_lvis_meta_v1()["thing_classes"]
    labels = [class_names[i] for i in classes]
    labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

    num_instances = len(boxes)

    np.random.seed(args.seed)
    assigned_colors = [random_color(maximum=255) for _ in range(num_instances)]

    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    if areas is not None:
        sorted_idxs = np.argsort(-areas).tolist()
        # Re-order overlapped instances in descending order.
        boxes = boxes[sorted_idxs]
        labels = [labels[k] for k in sorted_idxs]
        masks = [masks[idx] for idx in sorted_idxs]
        assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

    default_font_size = int(max(np.sqrt(height * width) // 90, 10))

    for i in range(num_instances):
        color = assigned_colors[i]
        img_b = img.copy()

        # draw box
        x0, y0, x1, y1 = boxes[i]
        cv2.rectangle(
            img_b, (x0, y0), (x1, y1),
            color=color,
            thickness=default_font_size // 4)

        # draw segment
        polygons, _ = mask_to_polygons(masks[i])
        for points in polygons:
            points = np.array(points).reshape((1, -1, 2)).astype(np.int32)
            cv2.fillPoly(img_b, pts=[points], color=color)

        img = cv2.addWeighted(img, 0.5, img_b, 0.5, 0)

    for i in range(num_instances):
        color = assigned_colors[i]
        x0, y0, x1, y1 = boxes[i]

        SMALL_OBJECT_AREA_THRESH = 1000
        instance_area = (y1 - y0) * (x1 - x0)

        # for small objects, draw text at the side to avoid occlusion
        text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
        if instance_area < SMALL_OBJECT_AREA_THRESH or y1 - y0 < 40:
            if y1 >= height - 5:
                text_pos = (x1, y0)
            else:
                text_pos = (x0, y1)

        # draw label
        x, y = text_pos
        text = labels[i]
        font = cv2.FONT_HERSHEY_SIMPLEX
        height_ratio = (y1 - y0) / np.sqrt(height * width)
        font_scale = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5)
        font_thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, text_pos, (int(x + text_w * 0.6), y + text_h), (0, 0, 0), -1)
        cv2.putText(
            img, text, (x, y + text_h - 5),
            fontFace=font,
            fontScale=font_scale * 0.6,
            color=color_brightness(color, brightness_factor=0.7),
            thickness=font_thickness,
            lineType=cv2.LINE_AA)

    return img


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    img = img[:, :, ::-1]  # BGR -> RGB

    size = IMAGE_SIZE
    max_size = IMAGE_MAX_SIZE
    scale = size / min(im_h, im_w)
    if im_h < im_w:
        oh, ow = size, scale * im_w
    else:
        oh, ow = scale * im_h, size
    if max(oh, ow) > max_size:
        scale = max_size / max(oh, ow)
        oh = oh * scale
        ow = ow * scale
    ow = int(ow + 0.5)
    oh = int(oh + 0.5)

    img = np.asarray(Image.fromarray(img).resize((ow, oh), Image.BILINEAR))

    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(
        pred_boxes, scores, pred_classes, pred_masks, im_hw, pred_hw):
    scale_x, scale_y = (
        im_hw[1] / pred_hw[1],
        im_hw[0] / pred_hw[0],
    )

    pred_boxes[:, 0::2] *= scale_x
    pred_boxes[:, 1::2] *= scale_y
    pred_boxes[:, [0, 2]] = np.clip(pred_boxes[:, [0, 2]], 0, im_hw[1])
    pred_boxes[:, [1, 3]] = np.clip(pred_boxes[:, [1, 3]], 0, im_hw[0])

    threshold = 0
    widths = pred_boxes[:, 2] - pred_boxes[:, 0]
    heights = pred_boxes[:, 3] - pred_boxes[:, 1]
    keep = (widths > threshold) & (heights > threshold)

    pred_boxes = pred_boxes[keep]
    scores = scores[keep]
    pred_classes = pred_classes[keep]
    pred_masks = pred_masks[keep]

    mask_threshold = 0.5
    pred_masks = paste_masks_in_image(
        pred_masks[:, 0, :, :], pred_boxes,
        (im_hw[0], im_hw[1]), mask_threshold
    )

    pred = {
        'pred_boxes': pred_boxes,
        'scores': scores,
        'pred_classes': pred_classes,
        'pred_masks': pred_masks,
    }
    return pred


def predict(net, img):
    im_h, im_w = img.shape[:2]
    img = preprocess(img)
    pred_hw = img.shape[-2:]

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'img': img})

    pred_boxes, scores, pred_classes, pred_masks = output

    pred = post_processing(
        pred_boxes, scores, pred_classes, pred_masks,
        (im_h, im_w), pred_hw
    )

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

        res_img = draw_predictions(img, pred)

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

        # # show
        # cv2.imshow('frame', res_img)
        #
        # # save results
        # if writer is not None:
        #     writer.write(res_img.astype(np.uint8))

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