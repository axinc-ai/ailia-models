import numpy as np
import cv2
import json

color_palette = [(136, 112, 246),
                 (49, 136, 219),
                 (49, 156, 173),
                 (49, 170, 119),
                 (122, 176, 51),
                 (164, 172, 53),
                 (197, 168, 56),
                 (244, 154, 110),
                 (244, 121, 204),
                 (204, 101, 245)]  # husl


def decode_np(heatmap, scale, stride, default_pt, method='exp'):
    '''
    :param heatmap: [pt_num, h, w]
    :param scale:
    :return:
    '''
    kp_num, h, w = heatmap.shape
    dfx, dfy = np.array(default_pt) * scale / stride
    for k, hm in enumerate(heatmap):
        heatmap[k] = cv2.GaussianBlur(hm, (5, 5), 1)
    if method == 'exp':
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        heatmap_th = np.copy(heatmap)
        heatmap_th[heatmap < np.amax(heatmap) / 2] = 0
        heat_sums_th = np.sum(heatmap_th, axis=(1, 2))
        x = np.sum(heatmap_th * xx, axis=(1, 2))
        y = np.sum(heatmap_th * yy, axis=(1, 2))
        x = x / heat_sums_th
        y = y / heat_sums_th
        x[heat_sums_th == 0] = dfx
        y[heat_sums_th == 0] = dfy
    else:
        if method == 'max':
            heatmap_th = heatmap.reshape(kp_num, -1)
            y, x = np.unravel_index(np.argmax(heatmap_th, axis=1), [h, w])
        elif method == 'maxoffset':
            heatmap_th = heatmap.reshape(kp_num, -1)
            si = np.argsort(heatmap_th, axis=1)
            y1, x1 = np.unravel_index(si[:, -1], [h, w])
            y2, x2 = np.unravel_index(si[:, -2], [h, w])
            x = (3 * x1 + x2) / 4.
            y = (3 * y1 + y2) / 4.
        var = np.var(heatmap_th, axis=1)
        x[var < 1] = dfx
        y[var < 1] = dfy
    x = x * stride / scale
    y = y * stride / scale
    return np.rint(x + 2), np.rint(y + 2)


def draw_keypoints(image, keypoints, gt_keypoints=None):
    '''
    :param image:
    :param keypoints: [[x, y, v], ...]
    :return:
    '''
    alpha = 0.8
    color1 = (0, 255, 0)
    color2 = (0, 0, 255)
    thick = 2
    l = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    overlay = image.copy()
    if gt_keypoints is None:
        for i, kpt in enumerate(keypoints):
            x, y, v = kpt
            if v > 0:
                overlay = cv2.line(
                    overlay, (x - l, y - l), (x + l, y + l),
                    color_palette[i % len(color_palette)],
                    thick)
                overlay = cv2.line(
                    overlay, (x - l, y + l), (x + l, y - l),
                    color_palette[i % len(color_palette)],
                    thick)

    if gt_keypoints is not None:
        for k in range(len(keypoints)):
            gtx, gty, gtv = gt_keypoints[k]
            x, y, v = keypoints[k]
            if gtv > 0:
                overlay = cv2.line(overlay, (x - l, y - l), (x + l, y + l), color1, thick)
                overlay = cv2.line(overlay, (x - l, y + l), (x + l, y - l), color1, thick)
                overlay = cv2.putText(overlay, str(k), (x, y), font, font_scale, color1, thick, cv2.LINE_AA)
                overlay = cv2.line(overlay, (gtx - l, gty - l), (gtx + l, gty + l), color2, thick)
                overlay = cv2.line(overlay, (gtx - l, gty + l), (gtx + l, gty - l), color2, thick)
                overlay = cv2.putText(overlay, str(k), (gtx, gty), font, font_scale, color2, thick, cv2.LINE_AA)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image


def save_json(json_path, keypoints):
    results = []

    for kpt in keypoints:
        x, y, v = kpt.tolist()
        if v > 0:
            results.append({'x': x, 'y': y, 'v': v})

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
