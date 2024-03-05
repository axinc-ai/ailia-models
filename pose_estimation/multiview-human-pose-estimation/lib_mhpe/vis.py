# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

from lib_mhpe.inference import get_max_preds


def show_heatmaps(batch_image, batch_raw_image, batch_heatmaps, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.copy()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image = np.add(batch_image, -min)
        batch_image = np.divide(batch_image, (max - min + 1e-5))

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps)

    for i in range(batch_size):
        image = np.multiply(batch_image[i], 255)
        image = np.clip(image, 0, 255)
        image = np.transpose(image, (1, 2, 0))
        image = image.astype(np.uint8)

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        heatmaps = np.multiply(batch_heatmaps[i], 255)
        heatmaps = np.clip(heatmaps, 0, 255)
        heatmaps = heatmaps.astype(np.uint8)

        preds_raw = preds[i] / 64
        preds_raw[:, 0] = preds_raw[:, 0] * batch_raw_image[i].shape[2]
        preds_raw[:, 1] = preds_raw[:, 1] * batch_raw_image[i].shape[1]
        raw_image = batch_raw_image[i]
        raw_image = np.transpose(raw_image, (1, 2, 0))
        raw_image = raw_image.astype(np.uint8)

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])), 1,
                       [0, 0, 255], 1)
            cv2.circle(raw_image,
                       (int(preds_raw[j][0]), int(preds_raw[j][1])), 1,
                       [0, 0, 255], 5)

            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3

            cv2.circle(masked_image, (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    return grid_image, resized_image, raw_image
