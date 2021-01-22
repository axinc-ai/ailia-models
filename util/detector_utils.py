import os
import sys
import numpy as np
import cv2


def preprocessing_img(img):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    return img


def load_image(image_path):
    if os.path.isfile(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        print(f'[ERROR] {image_path} not found.')
        sys.exit()
    return preprocessing_img(img)


def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)


def plot_results(detector, img, category, segm_masks=None, logging=True, det_shape = None):
    """
    :param detector: ailia.Detector, or list of ailia.DetectorObject
    :param img: ndarray data of image
    :param category: list of category_name
    :param segm_masks:
    :param logging: output log flg
    :return:
    """
    h, w = img.shape[0], img.shape[1]

    pad_x = pad_y = 0
    if det_shape != None:
        scale = np.max((h / det_shape[0], w / det_shape[1]))
        start = (det_shape[0:2] - np.array(img.shape[0:2]) / scale) // 2
        pad_x = start[1]*scale
        pad_y = start[0]*scale

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
                f'  category={obj.category}[ {category[obj.category]} ]'
            )
            print(f'  prob={obj.prob}')
            print(f'  x={obj.x}')
            print(f'  y={obj.y}')
            print(f'  w={obj.w}')
            print(f'  h={obj.h}')

        color = hsv_to_rgb(256 * obj.category / (len(category) + 1), 255, 255)
        colors.append(color)

    # draw segmentation area
    if segm_masks:
        for idx in range(count):
            mask = np.repeat(np.expand_dims(segm_masks[idx], 2), 3, 2).astype(np.bool)
            color = colors[idx][:3]
            fill = np.repeat(np.repeat([[color]], img.shape[0], 0), img.shape[1], 1)
            img[:, :, :3][mask] = img[:, :, :3][mask] * 0.7 + fill[mask] * 0.3

    # draw bounding box
    for idx in range(count):
        obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
        top_left = (int((w + pad_x*2) * obj.x - pad_x), int((h + pad_y*2) * obj.y - pad_y))
        bottom_right = (int((w + pad_x*2) * (obj.x + obj.w) - pad_x), int((h + pad_y*2) * (obj.y + obj.h) - pad_y))

        color = colors[idx]
        cv2.rectangle(img, top_left, bottom_right, color, 4)

    # draw label
    for idx in range(count):
        obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
        text_position = (int((w + pad_x*2) * obj.x - pad_x) + 4, int((h + pad_y*2) * (obj.y + obj.h) - pad_y - 8))
        fontScale = w / 512.0

        color = colors[idx]
        cv2.putText(
            img,
            category[obj.category],
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )
    return img
