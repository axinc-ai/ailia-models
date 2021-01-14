import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.special import expit

MIN_SIZE = 800
MAX_SIZE = 1600
size_divisibility = 32
PIXEL_MEAN = [103.53, 116.28, 123.675]
PIXEL_STD = [1.0, 1.0, 1.0]
NUM_CLASSES = 1
NUM_KERNELS = 256
SCORE_THR = 0.1
NUM_GRIDS = [40, 36, 24, 16, 12]
FPN_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
MASK_THR = 0.5
NMS_PRE = 500
NMS_TYPE = 'mask'
NMS_SIGMA = 2
NMS_KERNEL = 'gaussian'
UPDATE_THR = 0.05
MAX_PER_IMG = 100

def resize_pad(img):
    """ resize and pad image to be fed into model

    """
    h, w = img.shape[:2]

    if h < w:
        newh, neww = MIN_SIZE, MIN_SIZE / h * w
    else:
        newh, neww = MIN_SIZE / w * h, MIN_SIZE

    max_size = max(newh, neww)
    if max_size > MAX_SIZE:
        scale = MAX_SIZE / max_size
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    img_new = cv2.resize(img, (neww, newh))

    max_size = max(img_new.shape)
    if size_divisibility > 1:
        stride = size_divisibility
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size + (stride - 1)) // stride * stride

    if h < w:
        padh = 0
        padw = max_size - img_new.shape[1]
    else:
        padh = max_size - img_new.shape[0]
        padw = 0
    padht = padh // 2
    padhb = padh // 2 + padh % 2
    padwl = padw // 2
    padwr = padw // 2 + padw % 2

    img_new = cv2.copyMakeBorder(img_new, padht, padhb, padwl, padwr, cv2.BORDER_CONSTANT, (0, 0, 0))
    return img_new

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'

    Source: https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)

def point_nms(heat, kernel=2):
    # kernel must be 2
    hmax = np.expand_dims(pool2d(heat.squeeze(), kernel, 1, 1), (0, 1))
    keep = (hmax[:, :, :-1, :-1] == heat).astype(np.float)
    return heat * keep

def matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores, sigma=2.0, kernel='gaussian'):
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []

    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = seg_masks @ seg_masks.T
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay / soft nms
    delay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'linear':
        delay_matrix = (1 - delay_iou) / (1 - compensate_iou)
        delay_coefficient, _ = delay_matrix.min(0)
    else:
        delay_matrix = np.exp(-1 * sigma * (delay_iou ** 2))
        compensate_matrix = np.exp(-1 * sigma * (compensate_iou ** 2))
        delay_coefficient, _ = (delay_matrix / compensate_matrix).min(0)

    # update the score.
    cate_scores_update = cate_scores * delay_coefficient

    return cate_scores_update

def mask_nms(cate_labels, seg_masks, sum_masks, cate_scores, nms_thr=0.5):
    n_samples = len(cate_scores)
    if n_samples == 0:
        return []

    keep = np.ones(cate_scores.shape)
    seg_masks = seg_masks.astype(float)

    for i in range(n_samples - 1):
        if not keep[i]:
            continue
        mask_i = seg_masks[i]
        label_i = cate_labels[i]
        for j in range(i + 1, n_samples, 1):
            if not keep[j]:
                continue
            mask_j = seg_masks[j]
            label_j = cate_labels[j]
            if label_i != label_j:
                continue
            # overlaps
            inter = (mask_i * mask_j).sum()
            union = sum_masks[i] + sum_masks[j] - inter
            if union > 0:
                if inter / union > nms_thr:
                    keep[j] = False
            else:
                keep[j] = False
    return keep

def inference_single_image(
    cate_preds, kernel_preds, seg_preds, cur_size, ori_size
):
    scores = []
    pred_classes = []
    pred_masks = []
    pred_boxes = []

    # overall info.
    h, w = cur_size
    f_h, f_w = seg_preds.shape[-2:]
    ratio = np.ceil(h/f_h)
    upsampled_size_out = (int(f_h*ratio), int(f_w*ratio))

    # process.
    inds = (cate_preds > SCORE_THR)
    cate_scores = cate_preds[inds]
    if len(cate_scores) == 0:
        return scores, pred_classes, pred_masks, pred_boxes

    # cate_labels & kernel_preds
    inds = inds.nonzero()
    # cate_labels = inds[:, 1]
    cate_labels = inds[1]
    kernel_preds = kernel_preds[inds[0]]

    # trans vector.
    # size_trans = cate_labels.new_tensor(NUM_GRIDS).pow(2).cumsum(0)
    size_trans = np.power(np.array(NUM_GRIDS), 2).cumsum(0)
    strides = np.ones(size_trans[-1])

    n_stage = len(NUM_GRIDS)
    strides[:size_trans[0]] *= FPN_INSTANCE_STRIDES[0]
    for ind_ in range(1, n_stage):
        strides[size_trans[ind_ - 1]:size_trans[ind_]] *= FPN_INSTANCE_STRIDES[ind_]
    strides = strides[inds[0]]

    # mask encoding.
    N, I = kernel_preds.shape
    kernel_preds = kernel_preds.reshape(N, I, 1, 1)
    # import torch
    # import torch.nn.functional as F
    B, _, H, W = seg_preds.shape
    tmp = np.empty((B, N, H, W))
    for i in range(N):
        tmp[0, i] = np.sum(seg_preds[0] * kernel_preds[i], axis=0)
    seg_preds = expit(tmp.squeeze(0))
    # a = torch.tensor(seg_preds)
    # b = torch.tensor(kernel_preds)
    # seg_preds = F.conv2d(a, b, stride=1)
    # seg_preds = seg_preds.squeeze(0).sigmoid()

    # mask.
    seg_masks = seg_preds > MASK_THR
    sum_masks = seg_masks.sum((1, 2)).astype(float)

    # filter.
    keep = sum_masks > strides
    if keep.sum() == 0:
        return scores, pred_classes, pred_masks, pred_boxes

    seg_masks = seg_masks[keep, ...]
    seg_preds = seg_preds[keep, ...]
    sum_masks = sum_masks[keep]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]

    # mask scoring.
    seg_scores = (seg_preds * seg_masks.astype(float)).sum((1, 2)) / sum_masks
    cate_scores *= seg_scores

    # sort and keep top nms_pre
    sort_inds = np.argsort(-cate_scores)
    if len(sort_inds) > NMS_PRE:
        sort_inds = sort_inds[:NMS_PRE]
    seg_masks = seg_masks[sort_inds, :, :]
    seg_preds = seg_preds[sort_inds, :, :]
    sum_masks = sum_masks[sort_inds]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    if NMS_TYPE == "matrix":
        # matrix nms & filter.
        cate_scores = matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                                        sigma=NMS_SIGMA, kernel=NMS_KERNEL)
        keep = cate_scores >= UPDATE_THR
    elif NMS_TYPE == "mask":
        # original mask nms.
        keep = mask_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                                nms_thr=MASK_THR)
    else:
        raise NotImplementedError

    if keep.sum() == 0:
        return scores, pred_classes, pred_masks, pred_boxes

    keep = keep.astype(bool)
    seg_preds = seg_preds[keep, :, :]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]

    # sort and keep top_k
    sort_inds = np.argsort(-cate_scores)
    if len(sort_inds) > MAX_PER_IMG:
        sort_inds = sort_inds[:MAX_PER_IMG]
    seg_preds = seg_preds[sort_inds, :, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    # reshape to original size.
    # import torch
    # import torch.nn.functional as F
    # a = torch.tensor(seg_preds).unsqueeze(0)
    # seg_preds = F.interpolate(a,
    #                             size=upsampled_size_out,
    #                             mode='bilinear')[:, :, :h, :w]
    C, _, _ = seg_preds.shape
    H, W = upsampled_size_out
    tmp = np.empty((C, H, W))
    for i in range(C):
        tmp[i] = cv2.resize(seg_preds[i], (W, H))
    seg_preds = tmp
    # seg_masks = F.interpolate(seg_preds,
    #                             size=ori_size,
    #                             mode='bilinear').squeeze(0)
    H, W = ori_size
    tmp = np.empty((C, H, W))
    for i in range(C):
        tmp[i] = cv2.resize(seg_preds[i], (W, H))
    seg_masks = tmp
    seg_masks = seg_masks > MASK_THR

    pred_classes = cate_labels
    scores = cate_scores
    pred_masks = seg_masks

    # get bbox from mask
    pred_boxes = np.zeros((seg_masks.shape[0], 4))
    #for i in range(seg_masks.size(0)):
    #    mask = seg_masks[i].squeeze()
    #    ys, xs = torch.where(mask)
    #    pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()       

    return scores, pred_classes, pred_masks, pred_boxes

def preprocess(img):
    img_new = resize_pad(img)
    pixel_mean = np.array(PIXEL_MEAN).reshape(1, 1, 3)
    pixel_std = np.array(PIXEL_STD).reshape(1, 1, 3)
    img_new = (img_new - pixel_mean) / pixel_std
    img_new = np.expand_dims(np.moveaxis(img_new, -1, 0), 0)[:, ::-1]

    return img_new

def postprocess(preds, cur_sizes, img_original):
    """
    docstring
    """
    cate_pred = preds[:5]
    kernel_pred = preds[5:10]
    mask_pred = preds[10]

    cate_pred = [np.moveaxis(point_nms(expit(cate_p), kernel=2), 1, -1)
                 for cate_p in cate_pred]

    results = []
    num_ins_levels = len(cate_pred)
    # image size.
    height, width = img_original.shape[:2]
    ori_size = (height, width)

    # prediction.
    pred_cate = [cate_pred[i][0].reshape(-1, NUM_CLASSES) for i in range(num_ins_levels)]
    pred_kernel = [np.moveaxis(kernel_pred[i][0], 0, -1).reshape(-1, NUM_KERNELS)
                   for i in range(num_ins_levels)]
    pred_mask = mask_pred

    pred_cate = np.concatenate(pred_cate, axis=0)
    pred_kernel = np.concatenate(pred_kernel, axis=0)
    print(pred_cate.shape, pred_kernel.shape, pred_mask.shape)

    # inference for single image.
    preds = inference_single_image(pred_cate, pred_kernel, pred_mask,
                                         cur_sizes, ori_size)
    # results.append({"instances": result})
    return preds