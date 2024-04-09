import cv2
import numpy as np

def hole_fill(img):
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_hole(parsing_result_filled, parsing_result, arm_mask):
    filled_hole = cv2.bitwise_and(np.where(parsing_result_filled == 4, 255, 0),
                                  np.where(parsing_result != 4, 255, 0)) - arm_mask * 255
    contours, _ = cv2.findContours(filled_hole, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    refine_hole_mask = np.zeros_like(parsing_result).astype(np.uint8)
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i], True)
        # keep hole > 2000 pixels
        if abs(a) > 2000:
            cv2.drawContours(refine_hole_mask, contours, i, color=255, thickness=-1)
    return refine_hole_mask + arm_mask

def refine_mask(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)
        # keep large area in skin case
        for j in range(len(area)):
          if j != i and area[i] > 2000:
             cv2.drawContours(refine_mask, contours, j, color=255, thickness=-1)
    return refine_mask

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette
