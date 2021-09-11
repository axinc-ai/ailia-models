"""
Copied and lightly modified from:
https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
"""

import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon

def roll_image(image):
    """helper for displaying images"""
    image = np.rollaxis(image, 0, 3)
    image -= image.min()
    image /= image.max()
    image *= 255
    return image.astype(np.uint8)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, masks, class_ids, class_names, figsize=(16, 16), ax=None,
                      show_mask=True, colors=None):

    # Number of instances
    N = len(masks)
    assert len(masks) == len(class_names)

    # If no axis is passed, create one and automatically call show()
    auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i, mask in enumerate(masks):
        if not np.any(mask):
            continue

        color = colors[i]

        # Label
        y, x = np.where(mask > 0)
        y1 = np.median(y)
        x1 = np.median(x)
        ax.text(x1, y1 + 8, class_names[i],
                color='w', size=14, backgroundcolor="none")

        # Mask
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
