import os
import sys
from logging import getLogger

import ailia
import cv2
import numpy as np

logger = getLogger(__name__)

sys.path.append(os.path.dirname(__file__))
from image_utils import imread  # noqa: E402


def preprocessing_img(img):
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    return img


def load_image(image_path):
    if os.path.isfile(image_path):
        img = imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        logger.error(f'{image_path} not found.')
        sys.exit()
    return preprocessing_img(img)


def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)


def letterbox_convert(frame, det_shape):
    """
    Adjust the size of the frame from the webcam to the ailia input shape.

    Parameters
    ----------
    frame: numpy array
    det_shape: tuple
        ailia model input (height,width)

    Returns
    -------
    resized_img: numpy array
        Resized `img` as well as adapt the scale
    """
    height, width = det_shape[0], det_shape[1]
    f_height, f_width = frame.shape[0], frame.shape[1]
    scale = np.max((f_height / height, f_width / width))

    # padding base
    img = np.zeros(
        (int(round(scale * height)), int(round(scale * width)), 3),
        np.uint8
    )
    start = (np.array(img.shape) - np.array(frame.shape)) // 2
    img[
        start[0]: start[0] + f_height,
        start[1]: start[1] + f_width
    ] = frame
    resized_img = cv2.resize(img, (width, height))
    return resized_img


def reverse_letterbox(detections, img, det_shape):
    h, w = img.shape[0], img.shape[1]

    pad_x = pad_y = 0
    if det_shape != None:
        scale = np.max((h / det_shape[0], w / det_shape[1]))
        start = (det_shape[0:2] - np.array(img.shape[0:2]) / scale) // 2
        pad_x = start[1] * scale
        pad_y = start[0] * scale

    new_detections = []
    for detection in detections:
        logger.debug(detection)
        r = ailia.DetectorObject(
            category=detection.category,
            prob=detection.prob,
            x=(detection.x * (w + pad_x * 2) - pad_x) / w,
            y=(detection.y * (h + pad_y * 2) - pad_y) / h,
            w=(detection.w * (w + pad_x * 2)) / w,
            h=(detection.h * (h + pad_y * 2)) / h,
        )
        new_detections.append(r)

    return new_detections


def plot_results(detector, img, category=None, segm_masks=None, logging=True):
    """
    :param detector: ailia.Detector, or list of ailia.DetectorObject
    :param img: ndarray data of image
    :param category: list of category_name
    :param segm_masks:
    :param logging: output log flg
    :return:
    """
    h, w = img.shape[0], img.shape[1]

    count = detector.get_object_count() if hasattr(detector, 'get_object_count') else len(detector)
    if logging:
        print(f'object_count={count}')

    # prepare color data
    colors = []
    for idx in range(count):
        obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]

        # print result
        if logging:
            print(f'+ idx={idx}')
            print(
                f'  category={obj.category}[ {category[int(obj.category)]} ]'
                if not isinstance(obj.category, str) and category is not None
                else f'  category=[ {obj.category} ]'
            )
            print(f'  prob={obj.prob}')
            print(f'  x={obj.x}')
            print(f'  y={obj.y}')
            print(f'  w={obj.w}')
            print(f'  h={obj.h}')

        if isinstance(obj.category, int) and category is not None:
            color = hsv_to_rgb(256 * obj.category / (len(category) + 1), 255, 255)
        else:
            color = hsv_to_rgb(256 * idx / (len(detector) + 1), 255, 255)
        colors.append(color)

    # draw segmentation area
    if segm_masks is not None and 0 < len(segm_masks):
        for idx in range(count):
            mask = np.repeat(np.expand_dims(segm_masks[idx], 2), 3, axis=2).astype(bool)
            color = colors[idx][:3]
            fill = np.repeat(np.repeat([[color]], img.shape[0], 0), img.shape[1], 1)
            img[:, :, :3][mask] = img[:, :, :3][mask] * 0.7 + fill[mask] * 0.3

    # draw bounding box
    for idx in range(count):
        obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
        top_left = (int(w * obj.x), int(h * obj.y))
        bottom_right = (int(w * (obj.x + obj.w)), int(h * (obj.y + obj.h)))

        color = colors[idx]
        cv2.rectangle(img, top_left, bottom_right, color, 4)

    # draw label
    for idx in range(count):
        obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
        fontScale = w / 2048

        text = category[int(obj.category)] \
            if not isinstance(obj.category, str) and category is not None \
            else obj.category
        text = "{} {}".format(text, int(obj.prob * 100) / 100)
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)[0]
        tw = textsize[0]
        th = textsize[1]

        margin = 3

        x1 = int(w * obj.x)
        y1 = int(h * obj.y)
        x2 = x1 + tw + margin
        y2 = y1 + th + margin

        # check the x,y x2,y2 are inside the image
        if x1 < 0:
            x1 = 0
        elif x2 > w:
            x1 = w - (tw + margin)

        if y1 < 0:
            y1 = 0
        elif y2 > h:
            y1 = h - (th + margin)

        # recompute x2, y2 if shift occured
        x2 = x1 + tw + margin
        y2 = y1 + th + margin

        top_left = (x1, y1)
        bottom_right = (x2, y2)

        color = colors[idx]
        cv2.rectangle(img, top_left, bottom_right, color, thickness=-1)

        text_color = (255, 255, 255, 255)
        cv2.putText(
            img,
            text,
            (top_left[0], top_left[1] + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            text_color,
            1
        )
    return img


def write_predictions(file_name, detector, img=None, category=None):
    h, w = (img.shape[0], img.shape[1]) if img is not None else (1, 1)

    count = detector.get_object_count() if hasattr(detector, 'get_object_count') else len(detector)

    with open(file_name, 'w') as f:
        for idx in range(count):
            obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
            label = category[int(obj.category)] \
                if not isinstance(obj.category, str) and category is not None \
                else obj.category

            f.write('%s %f %d %d %d %d\n' % (
                label.replace(' ', '_'),
                obj.prob,
                int(w * obj.x), int(h * obj.y),
                int(w * obj.w), int(h * obj.h),
            ))
