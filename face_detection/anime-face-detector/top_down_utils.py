import cv2
import numpy as np

from post_transforms_utils import transform_preds


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        heatmap height: H
        heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
                heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
                heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
                heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (
                heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
                heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray[N, K, H, W]: Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def keypoints_from_heatmaps(
        heatmaps, center, scale,
        kernel=11):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        batch size: N
        num keypoints: K
        heatmap height: H
        heatmap width: W
    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    # Avoid being affected
    heatmaps = heatmaps.copy()

    N, K, H, W = heatmaps.shape

    preds, maxvals = _get_max_preds(heatmaps)
    # apply Gaussian distribution modulation.
    heatmaps = np.log(
        np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
    for n in range(N):
        for k in range(K):
            preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [W, H], use_udp=False)

    return preds, maxvals
