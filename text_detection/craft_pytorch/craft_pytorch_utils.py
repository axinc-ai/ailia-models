import math

import numpy as np
import cv2
from skimage import io
import json


def load_image(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img


def pre_process(image):
    img_resized, target_ratio, _ = resize_aspect_ratio(
        image, 1280, 1.5, cv2.INTER_LINEAR
    )
    ratio_h = ratio_w = 1 / target_ratio
    x = normalize_mean_variance(img_resized)
    x = x.transpose(2, 0, 1)
    x = np.expand_dims(x, 0)

    return x, ratio_w, ratio_h


def post_process(y, image, ratio_w, ratio_h, json_path=None):
    score_text = y[0, :, :, 0]
    score_link = y[0, :, :, 1]
    boxes, polys = get_det_boxes(
        score_text,
        score_link,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        poly=False,
    )
    boxes = adjust_result_coordinates(boxes, ratio_w, ratio_h)
    polys = adjust_result_coordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    if json_path is not None:
        save_result_json(json_path, polys)

    return save_result(image[:, :, ::-1], polys)


# ======================
# Utils
# ======================

def resize_aspect_ratio(img, square_size, mag_ratio, interpolation):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap


def normalize_mean_variance(
        in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([
        mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0
    ], dtype=np.float32)
    img /= np.array([
        variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0
    ], dtype=np.float32)
    return img


def get_det_boxes(
        textmap, linkmap, text_threshold, link_threshold, low_text, poly=False
):
    boxes, labels, mapper = get_det_boxes_core(
        textmap, linkmap, text_threshold, link_threshold, low_text
    )

    if poly:
        polys = get_poly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys


def warp_coord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])


def get_det_boxes_core(
        textmap, linkmap, text_threshold, link_threshold, low_text
):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        # remove link area
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = \
            x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(
            np.array(np.where(segmap != 0)), 1, axis=0
        ).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def get_poly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []
    for k, box in enumerate(boxes):
        # size filter for small instance
        w = int(np.linalg.norm(box[0] - box[1]) + 1)
        h = int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 10 or h < 10:
            polys.append(None)
            continue

        # warp image
        tar = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(
            labels, M, (w, h), flags=cv2.INTER_NEAREST
        )
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None)
            continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:, i] != 0)[0]
            if len(region) < 2:
                continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len:
                max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None)
            continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     # segment width
        pp = [None] * num_cp    # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0, len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0:
                    break
                cp_section[seg_num] = [
                    cp_section[seg_num][0] / num_sec,
                    cp_section[seg_num][1] / num_sec
                ]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [
                cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy
            ]
            num_sec += 1

            if seg_num % 2 == 0:
                continue  # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1)/2)] = (x, cy)
                seg_height[int((seg_num - 1)/2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [
                cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec
            ]

        # pass if num of pivots is not sufficient or segment widh is
        # smaller than character height
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None)
            continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:     # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + \
            (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + \
            (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(
                    line_img,
                    (int(p[0]), int(p[1])),
                    (int(p[2]), int(p[3])),
                    1,
                    thickness=1,
                )
                if np.sum(np.logical_and(word_label, line_img)) == 0 or \
                   r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(
                    line_img,
                    (int(p[0]), int(p[1])),
                    (int(p[2]), int(p[3])),
                    1,
                    thickness=1,
                )
                if np.sum(np.logical_and(word_label, line_img)) == 0 or \
                   r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None)
            continue

        # make final polygon
        poly = []
        poly.append(warp_coord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warp_coord(Minv, (p[0], p[1])))
        poly.append(warp_coord(Minv, (epp[0], epp[1])))
        poly.append(warp_coord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warp_coord(Minv, (p[2], p[3])))
        poly.append(warp_coord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys


def adjust_result_coordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


def save_result(img, boxes):
    img = np.array(img)
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        # strResult = ','.join([str(p) for p in poly]) + '\r\n'

        poly = poly.reshape(-1, 2)
        cv2.polylines(
            img,
            [poly.reshape((-1, 1, 2))],
            True,
            color=(0, 0, 255),
            thickness=2,
        )
        # ptColor = (0, 255, 255)

    return img


def save_result_json(json_path, boxes):
    results = []
    for box in boxes:
        points = []
        for point in box:
            points.append(float(point[0]))
            points.append(float(point[1]))
        results.append(points)
    with open(json_path, 'w') as f:
        json.dump({"boxes": results}, f, indent=2)
