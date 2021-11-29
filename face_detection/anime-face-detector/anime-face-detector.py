import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from image_utils import normalize_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

from post_transforms_utils import flip_back, get_affine_transform
from top_down_utils import keypoints_from_heatmaps

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_YOLOV3_PATH = 'anime-face_yolov3.onnx'
MODEL_YOLOV3_PATH = 'anime-face_yolov3.onnx.prototxt'
WEIGHT_FASTERRCNN_PATH = 'anime-face_faster-rcnn.onnx'
MODEL_FASTERRCNN_PATH = 'anime-face_faster-rcnn.onnx.prototxt'
WEIGHT_LANDMARK_PATH = 'anime-face_hrnetv2.onnx'
MODEL_LANDMARK_PATH = 'anime-face_hrnetv2.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/anime-face-detector/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Anime Face Detector', IMAGE_PATH, SAVE_IMAGE_PATH
)
# parser.add_argument(
#     '-d', '--detection',
#     action='store_true',
#     help='Use object detection.'
# )
parser.add_argument(
    '-d', '--detector', default=None, choices=('yolov3', 'faster-rcnn'),
    help='face detector model.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def update_pred_box(pred_boxes):
    box_scale_factor = 1.1

    boxes = []
    for pred_box in pred_boxes:
        box = pred_box[:4]
        size = box[2:] - box[:2] + 1
        new_size = size * box_scale_factor
        center = (box[:2] + box[2:]) / 2
        tl = center - new_size / 2
        br = tl + new_size
        pred_box[:4] = np.concatenate([tl, br])
        boxes.append(pred_box)

    return boxes


def xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0] + 1
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1] + 1

    return bbox_xywh


# ======================
# Main functions
# ======================

def preprocess(img, resize_shape):
    h, w = resize_shape
    im_h, im_w, _ = img.shape

    # adaptive_resize
    scale = h / max(im_h, im_w)
    ow, oh = int(im_w * scale), int(im_h * scale)
    if ow != im_w or oh != im_h:
        _img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

        img = np.zeros((h, w, 3), dtype=np.uint8)
        ph, pw = (h - oh) // 2, (w - ow) // 2
        img[ph: ph + oh, pw: pw + ow] = _img
    else:
        ph = pw = 0

    img = img[:, :, ::-1]  # GBR -> RGB
    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    # return img
    return img, (ph, pw), (oh, ow)


def box2cs(box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """
    input_size = (256, 256)

    x, y, w, h = box[:4]
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale


def detect_faces(img, face_detector):
    im_h, im_w = img.shape[:2]

    # shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    shape = (608, 608)
    img, pad_hw, resized_hw = preprocess(img, shape)

    # feedforward
    output = face_detector.predict([img])
    boxes, _ = output
    boxes = boxes[0]
    boxes = boxes[boxes[:, 4] > 0]

    pad_x = pad_hw[1]
    pad_y = pad_hw[0]
    resized_x = resized_hw[1]
    resized_y = resized_hw[0]
    boxes[:, 0] = boxes[:, 0] - pad_y
    boxes[:, 2] = boxes[:, 2] - pad_x
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * im_h / resized_y
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * im_w / resized_x

    # scale boxes
    boxes = update_pred_box(boxes)

    return boxes


def keypoint_head_decode(output, img_metas):
    """Decode keypoints from heatmaps.

    Args:
        img_metas (list(dict)): Information about data augmentation
            By default this includes:
            - "image_file: path to the image file
            - "center": center of the bbox
            - "scale": scale of the bbox
            - "rotation": rotation of the bbox
            - "bbox_score": score of bbox
        output (np.ndarray[N, K, H, W]): model predicted heatmaps.
    """
    batch_size = len(output)

    c = np.zeros((batch_size, 2), dtype=np.float32)
    s = np.zeros((batch_size, 2), dtype=np.float32)
    for i in range(batch_size):
        c[i, :] = img_metas[i]['center']
        s[i, :] = img_metas[i]['scale']

    preds, maxvals = keypoints_from_heatmaps(
        output, c, s,
        kernel=11)

    all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals

    return all_preds


def predict(landmark_detector, face_detector, img):
    if face_detector is not None:
        bboxes = detect_faces(img, face_detector)
    else:
        h, w = img.shape[:2]
        bboxes = [np.array([0, 0, w - 1, h - 1, 1])]

    pose_results = []

    if len(bboxes) == 0:
        return pose_results

    bboxes = np.array(bboxes)
    bboxes_xywh = xyxy2xywh(bboxes)

    img_size = (256, 256)
    batch_data = []
    img_metas = []
    for bbox in bboxes_xywh:
        c, s = box2cs(bbox)
        r = 0
        img_metas.append({
            "center": c,
            "scale": s,
        })
        trans = get_affine_transform(c, s, r, img_size)
        _img = cv2.warpAffine(
            img,
            trans, (img_size[0], img_size[1]),
            flags=cv2.INTER_LINEAR)

        _img = normalize_image(_img[:, :, ::-1], 'ImageNet')
        batch_data.append(_img)

    batch_data = np.asarray(batch_data)
    batch_data = batch_data.transpose((0, 3, 1, 2))

    output = landmark_detector.predict([batch_data])
    heatmap = output[0]
    if 1:  # do flip
        batch_data = batch_data[:, :, :, ::-1]  # horizontal flip
        output = landmark_detector.predict([batch_data])
        flipped_heatmap = output[0]

        flip_pairs = [
            [0, 4], [1, 3], [5, 10], [6, 9],
            [7, 8], [11, 19], [12, 18], [13, 17],
            [14, 22], [15, 21], [16, 20], [24, 26]]
        flipped_heatmap = flip_back(
            flipped_heatmap,
            flip_pairs)

        # feature is not aligned, shift flipped heatmap for higher accuracy
        flipped_heatmap[:, :, :, 1:] = flipped_heatmap[:, :, :, :-1]

        heatmap = (heatmap + flipped_heatmap) * 0.5

    keypoint_result = keypoint_head_decode(heatmap, img_metas)

    return keypoint_result


def recognize_from_image(landmark_detector, face_detector):
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
                preds = predict(landmark_detector, face_detector, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            preds = predict(landmark_detector, face_detector, img)

        # # plot result
        # savepath = get_savepath(args.savepath, image_path, ext='.png')
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(landmark_detector, face_detector):
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
        preds = predict(landmark_detector, face_detector, img)

        # # plot result
        # res_img = draw_bbox(frame, bboxes)
        #
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
    logger.info('Checking detect_landmarks model...')
    check_and_download_models(WEIGHT_LANDMARK_PATH, MODEL_LANDMARK_PATH, REMOTE_PATH)

    model_path = weight_path = None
    if args.video or args.detector:
        args.detector = args.detector if args.detector else 'yolov3'

        dic_model = {
            'yolov3': (WEIGHT_YOLOV3_PATH, MODEL_YOLOV3_PATH),
            'faster-rcnn': (WEIGHT_LANDMARK_PATH, MODEL_LANDMARK_PATH),
        }
        weight_path, model_path = dic_model[args.detector]

        logger.info('Check face_detector model...')
        check_and_download_models(
            weight_path, model_path, REMOTE_PATH
        )

    env_id = args.env_id

    # initialize
    landmark_detector = ailia.Net(
        MODEL_LANDMARK_PATH, WEIGHT_LANDMARK_PATH, env_id=env_id)
    if model_path:
        face_detector = ailia.Net(model_path, weight_path, env_id=env_id)
    else:
        face_detector = None

    if args.video is not None:
        recognize_from_video(landmark_detector, face_detector)
    else:
        recognize_from_image(landmark_detector, face_detector)


if __name__ == '__main__':
    main()
