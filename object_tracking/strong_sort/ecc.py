# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 3/11/2019

import cv2
import numpy as np


def ECC(src, dst, warp_mode=cv2.MOTION_EUCLIDEAN, eps=1e-5,
        max_iter=100, scale=0.1, align=False):
    """Compute the warp matrix from src to dst.

    Parameters
    ----------
    src : ndarray
        An NxM matrix of source img(BGR or Gray), it must be the same format as dst.
    dst : ndarray
        An NxM matrix of target img(BGR or Gray).
    warp_mode: flags of opencv
        translation: cv2.MOTION_TRANSLATION
        rotated and shifted: cv2.MOTION_EUCLIDEAN
        affine(shift,rotated,shear): cv2.MOTION_AFFINE
        homography(3d): cv2.MOTION_HOMOGRAPHY
    eps: float
        the threshold of the increment in the correlation coefficient between two iterations
    max_iter: int
        the number of iterations.
    scale: float or [int, int]
        scale_ratio: float
        scale_size: [W, H]
    align: bool
        whether to warp affine or perspective transforms to the source image

    Returns
    -------
    warp matrix : ndarray
        Returns the warp matrix from src to dst.
        if motion models is homography, the warp matrix will be 3x3, otherwise 2x3
    src_aligned: ndarray
        aligned source image of gray
    """
    assert src.shape == dst.shape, "the source image must be the same format to the target image!"

    # BGR2GRAY
    if src.ndim == 3:
        # Convert images to grayscale
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # make the imgs smaller to speed up
    if scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            if scale != 1:
                src_r = cv2.resize(src, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                scale = [scale, scale]
            else:
                src_r, dst_r = src, dst
                scale = None
        else:
            if scale[0] != src.shape[1] and scale[1] != src.shape[0]:
                src_r = cv2.resize(src, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                scale = [scale[0] / src.shape[1], scale[1] / src.shape[0]]
            else:
                src_r, dst_r = src, dst
                scale = None
    else:
        src_r, dst_r = src, dst

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (cc, warp_matrix) = cv2.findTransformECC(src_r, dst_r, warp_matrix, warp_mode, criteria, None, 1)
    except cv2.error:
        return None, None

    if scale is not None:
        warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
        warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

    if align:
        sz = src.shape
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            src_aligned = cv2.warpPerspective(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            src_aligned = cv2.warpAffine(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
        return warp_matrix, src_aligned
    else:
        return warp_matrix, None
