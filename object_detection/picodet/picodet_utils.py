import numpy as np

from math_utils import softmax, sigmoid
from nms_utils import batched_nms

prior_generator_strides = [
    (stride, stride) for stride in [8, 16, 32, 64]
]


def meshgrid(x, y, row_major=True):
    xx, yy = np.meshgrid(x, y)
    if row_major:
        return xx.reshape(-1), yy.reshape(-1)
    else:
        return yy.reshape(-1), xx.reshape(-1)


def single_level_grid_priors(
        featmap_size,
        level_idx,
        with_stride=False):
    """Generate grid Points of a single level.

    Note:
        This function is usually called by method ``self.grid_priors``.

    Args:
        featmap_size (tuple[int]): Size of the feature maps, arrange as
            (h, w).
        level_idx (int): The index of corresponding feature map level.
        with_stride (bool): Concatenate the stride to the last dimension
            of points.

    Return:
        Tensor: Points of single feature levels.
        The shape of tensor should be (N, 2) when with stride is
        ``False``, where N = width * height, width and height
        are the sizes of the corresponding feature level,
        and the last dimension 2 represent (coord_x, coord_y),
        otherwise the shape should be (N, 4),
        and the last dimension 4 represent
        (coord_x, coord_y, stride_w, stride_h).
    """
    strides = prior_generator_strides
    offset = 0.5

    feat_h, feat_w = featmap_size
    stride_w, stride_h = strides[level_idx]
    shift_x = (np.arange(0, feat_w) + offset) * stride_w
    shift_y = (np.arange(0, feat_h) + offset) * stride_h

    shift_xx, shift_yy = meshgrid(shift_x, shift_y)
    if not with_stride:
        shifts = np.stack([shift_xx, shift_yy], axis=-1)
    else:
        stride_w = np.full(
            (shift_xx.shape[0],), stride_w)
        stride_h = np.full(
            (shift_yy.shape[0],), stride_h)
        shifts = np.stack(
            [shift_xx, shift_yy, stride_w, stride_h],
            axis=-1)

    return shifts


def grid_priors(
        num_levels,
        featmap_sizes,
        with_stride=False):
    """Generate grid points of multiple feature levels.

    Args:
        num_levels:
        featmap_sizes (list[tuple]): List of feature map sizes in
            multiple feature levels, each size arrange as
            as (h, w).
        with_stride (bool): Whether to concatenate the stride to
            the last dimension of points.

    Return:
        list[torch.Tensor]: Points of  multiple feature levels.
        The sizes of each tensor should be (N, 2) when with stride is
        ``False``, where N = width * height, width and height
        are the sizes of the corresponding feature level,
        and the last dimension 2 represent (coord_x, coord_y),
        otherwise the shape should be (N, 4),
        and the last dimension 4 represent
        (coord_x, coord_y, stride_w, stride_h).
    """

    key = (num_levels, tuple(featmap_sizes))
    if key in grid_priors.multi_level_priors:
        return grid_priors.multi_level_priors[key]

    multi_level_priors = []
    for i in range(num_levels):
        priors = single_level_grid_priors(
            featmap_sizes[i],
            level_idx=i,
            with_stride=with_stride)
        multi_level_priors.append(priors)

    grid_priors.multi_level_priors[key] = multi_level_priors

    return multi_level_priors


grid_priors.multi_level_priors = {}


class Integral:
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.project = np.linspace(0, self.reg_max, self.reg_max + 1)

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = softmax(x.reshape(-1, self.reg_max + 1), axis=1)
        x = x.dot(self.project)
        x = x.reshape(-1, 4)
        return x


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = np.stack([x1, y1, x2, y2], axis=-1)

    if max_shape is not None:
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, max_shape[1])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, max_shape[0])
        return bboxes

    return bboxes


def filter_scores_and_topk(
        scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = np.nonzero(valid_mask)

    num_topk = min(topk, len(valid_idxs[0]))

    idxs = np.argsort(-scores)
    scores = scores[idxs]
    scores = scores[:num_topk]
    topk_idxs = idxs[:num_topk]
    keep_idxs = valid_idxs[0][topk_idxs]
    labels = valid_idxs[1][topk_idxs]

    filtered_results = {k: v[keep_idxs] for k, v in results.items()}

    return scores, labels, keep_idxs, filtered_results


def bbox_post_process(
        mlvl_scores,
        mlvl_labels,
        mlvl_bboxes,
        scale_factor=None,
        with_nms=True,
        nms_thre=0.6,
        mlvl_score_factors=None):
    """bbox post-processing method.

    The boxes would be rescaled to the original image scale and do
    the nms operation. Usually with_nms is False is used for aug test.

    Args:
        mlvl_scores (list[Tensor]): Box scores from all scale
            levels of a single image, each item has shape
            (num_bboxes, ).
        mlvl_labels (list[Tensor]): Box class labels from all scale
            levels of a single image, each item has shape
            (num_bboxes, ).
        mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
            levels of a single image, each item has shape (num_bboxes, 4).
        scale_factor (ndarray, optional): Scale factor of the image arange
            as (w_scale, h_scale, w_scale, h_scale).
        with_nms (bool): If True, do nms before return boxes.
            Default: True.
        mlvl_score_factors (list[Tensor], optional): Score factor from
            all scale levels of a single image, each item has shape
            (num_bboxes, ). Default: None.

    Returns:
        tuple[Tensor]: Results of detected bboxes and labels. If with_nms
            is False and mlvl_score_factor is None, return mlvl_bboxes and
            mlvl_scores, else return mlvl_bboxes, mlvl_scores and
            mlvl_score_factor. Usually with_nms is False is used for aug
            test. If with_nms is True, then return the following format

            - det_bboxes (Tensor): Predicted bboxes with shape \
                [num_bboxes, 5], where the first 4 columns are bounding \
                box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                column are scores between 0 and 1.
            - det_labels (Tensor): Predicted labels of the corresponding \
                box with shape [num_bboxes].
    """

    mlvl_bboxes = np.concatenate(mlvl_bboxes)
    if scale_factor is not None:
        mlvl_bboxes /= scale_factor

    mlvl_scores = np.concatenate(mlvl_scores)
    mlvl_labels = np.concatenate(mlvl_labels)

    if mlvl_score_factors is not None:
        # TODO： Add sqrt operation in order to be consistent with
        #  the paper.
        mlvl_score_factors = np.concatenate(mlvl_score_factors)
        mlvl_scores = mlvl_scores * mlvl_score_factors

    max_per_img = 100
    if with_nms:
        if len(mlvl_bboxes) == 0:
            det_bboxes = np.concatenate([mlvl_bboxes, mlvl_scores[:, None]], -1)
            return det_bboxes, mlvl_labels

        keep_idxs = batched_nms(
            mlvl_bboxes, mlvl_scores, mlvl_labels, nms_thre)

        det_bboxes = mlvl_bboxes[keep_idxs][:max_per_img]
        det_labels = mlvl_labels[keep_idxs][:max_per_img]

        scores = mlvl_scores[keep_idxs][:max_per_img].reshape(-1, 1)
        det_bboxes = np.concatenate([det_bboxes, scores], axis=1)

        return det_bboxes, det_labels
    else:
        return mlvl_bboxes, mlvl_scores, mlvl_labels


def get_bboxes(
        cls_score_list,
        bbox_pred_list,
        mlvl_priors,
        img_shape,
        cls_channels,
        scale_factor=None,
        with_nms=True,
        nms_thre=0.6,
        score_thr = 0.025):
    """Transform outputs of a single image into bbox predictions.

    Args:
        cls_score_list (list[Tensor]): Box scores from all scale
            levels of a single image, each item has shape
            (num_priors * num_classes, H, W).
        bbox_pred_list (list[Tensor]): Box energies / deltas from
            all scale levels of a single image, each item has shape
            (num_priors * 4, H, W).
        mlvl_priors (list[Tensor]): Each element in the list is
            the priors of a single level in feature pyramid, has shape
            (num_priors, 4).
        img_shape:
        cls_channels:
        scale_factor (ndarray, optional): Scale factor of the image arange
            as (w_scale, h_scale, w_scale, h_scale).
        with_nms (bool): If True, do nms before return boxes.
            Default: True.
        nms_thre:

    Returns:
        tuple[Tensor]: Results of detected bboxes and labels. If with_nms
            is False and mlvl_score_factor is None, return mlvl_bboxes and
            mlvl_scores, else return mlvl_bboxes, mlvl_scores and
            mlvl_score_factor. Usually with_nms is False is used for aug
            test. If with_nms is True, then return the following format

            - det_bboxes (Tensor): Predicted bboxes with shape \
                [num_bboxes, 5], where the first 4 columns are bounding \
                box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                column are scores between 0 and 1.
            - det_labels (Tensor): Predicted labels of the corresponding \
                box with shape [num_bboxes].
    """
    reg_max = 7
    integral = Integral(reg_max)

    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_labels = []
    for level_idx, (cls_score, bbox_pred, stride, priors) in \
            enumerate(zip(
                cls_score_list, bbox_pred_list,
                prior_generator_strides, mlvl_priors)):
        bbox_pred = bbox_pred.transpose(1, 2, 0)
        bbox_pred = integral.forward(bbox_pred) * stride[0]
        scores = cls_score.transpose(1, 2, 0).reshape(-1, cls_channels)
        scores = sigmoid(scores)

        # After https://github.com/open-mmlab/mmdetection/pull/6268/,
        # this operation keeps fewer bboxes under the same `nms_pre`.
        # There is no difference in performance for most models. If you
        # find a slight drop in performance, you can set a larger
        # `nms_pre` than before.
        nms_pre = 1000
        results = filter_scores_and_topk(
            scores, score_thr, nms_pre,
            dict(bbox_pred=bbox_pred, priors=priors))
        scores, labels, _, filtered_results = results

        bbox_pred = filtered_results['bbox_pred']
        priors = filtered_results['priors']

        bboxes = distance2bbox(
            priors, bbox_pred, max_shape=img_shape)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
        mlvl_labels.append(labels)

    return bbox_post_process(
        mlvl_scores,
        mlvl_labels,
        mlvl_bboxes,
        scale_factor,
        with_nms=with_nms,
        nms_thre=nms_thre)
