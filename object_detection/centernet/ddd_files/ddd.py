import numpy as np
import cv2
import onnxruntime

from .ddd_utils import ddd2locrot
from .image import *
from .ddd_model_constants import *

# ---------------------------------------------------
# Post-processing functions
# ---------------------------------------------------

def post_process(dets, meta, threshold=opt.peak_thresh, scale=1):
    # make copy of dets that can be modified and pick out center, scale, calibration values later used for transformation array
    dets = ddd_post_process_2d(dets.copy(), meta['c'], meta['s'], opt)
    dets = ddd_post_process_3d(dets, meta['calib'], threshold)
    return dets

def ddd_post_process_2d(dets, c, s, opt):
    # rescale detection points, depth, rotation, correction vector, etc to original image size for every class
    ret = []
    include_wh = dets.shape[1] > 16
    top_preds = {}
    dets[:, :2] = transform_preds(
          dets[:, 0:2], c, s, (opt.output_w, opt.output_h))
    classes = dets[:, -1]
    for j in range(opt.num_classes):
        inds = (classes == j)
        top_preds[j + 1] = np.concatenate([
                            dets[inds, :3].astype(np.float32),
                            get_alpha(dets[inds, 3:11])[:, np.newaxis].astype(np.float32),
                            get_pred_depth(dets[inds, 11:12]).astype(np.float32),
                            dets[inds, 12:15].astype(np.float32)], axis=1)
        if include_wh:
            top_preds[j + 1] = np.concatenate([
                top_preds[j + 1],
                transform_preds(dets[inds, 15:17], c, s, (opt.output_w, opt.output_h)).astype(np.float32)], axis=1)
    return top_preds

def get_pred_depth(depth):
    return depth

def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)

def ddd_post_process_3d(dets, calibs, threshold=opt.peak_thresh):
    # create 3d bounding boxes
    ret = []
    preds = {}
    for cls_ind in dets.keys():
        preds[cls_ind] = []
        for j in range(len(dets[cls_ind])):
            center = dets[cls_ind][j][:2]
            score = dets[cls_ind][j][2]
            alpha = dets[cls_ind][j][3]
            depth = dets[cls_ind][j][4]
            dimensions = dets[cls_ind][j][5:8]
            wh = dets[cls_ind][j][8:10]
            locations, rotation_y = ddd2locrot(
              center, alpha, dimensions, depth, calibs)
            bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                    center[0] + wh[0] / 2, center[1] + wh[1] / 2]
            pred = [alpha] + bbox + dimensions.tolist() + \
                   locations.tolist() + [rotation_y, score]
            preds[cls_ind].append(pred)
        preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    return merge_outputs(preds, threshold)

def merge_outputs(dets, threshold):
    for j in range(1, 4):
          if len(dets[j] > 0):
            keep_inds = (dets[j][:, -1] > threshold)
            dets[j] = dets[j][keep_inds]
    return dets

# ---------------------------------------------------
# Main processing functions
# ---------------------------------------------------

HEADS = ['hm', 'dep', 'rot', 'dim', 'wh', 'reg']
def process(image):
    # reformat list output into dictionary with original labels
    output_raw = forward(image)
    output = {label: out for label, out in zip(HEADS, output_raw)}
 
    output['hm'] = sigmoid(output['hm'])
    output['dep'] = 1. / (sigmoid(output['dep']) + 1e-6) - 1.
    
    dets = ddd_decode(output['hm'][0], output['rot'][0], output['dep'][0],
                      output['dim'][0], wh=output['wh'][0], reg=output['reg'][0], k=opt.K)
    return output, dets

ONNX_PATH = '../models/ddd_dlav0.onnx'
def forward(x):
    onnx_session = onnxruntime.InferenceSession(ONNX_PATH)
    in_net = { onnx_session.get_inputs()[0].name: x }
    onnx_out = onnx_session.run(None, in_net)
    return onnx_out

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def topk(scores, k=40):
    cat, _, width = scores.shape

    scores = scores.reshape((cat, -1))
    # Temporarily using sort to check output equality to reference
#     topk_inds = np.argpartition(scores, -k, axis=1)[:, -k:]
    topk_inds = np.argsort(scores, axis=1)[:, -k:][...,::-1]
    topk_scores = scores[np.arange(scores.shape[0])[:, None], topk_inds]

    topk_ys = (topk_inds / width).astype(np.int32).astype(np.float)
    topk_xs = (topk_inds % width).astype(np.int32).astype(np.float)

    topk_scores = topk_scores.reshape((-1))
#     topk_ind = np.argpartition(topk_scores, -k)[-k:]
    topk_ind = np.argsort(topk_scores)[-k:][...,::-1]
    topk_score = topk_scores[topk_ind]
    topk_classes = (topk_ind / k).astype(np.int32)
    topk_inds = gather_feat(
        topk_inds.reshape((-1, 1)), topk_ind).reshape((k))
    topk_ys = gather_feat(topk_ys.reshape((-1, 1)), topk_ind).reshape((k))
    topk_xs = gather_feat(topk_xs.reshape((-1, 1)), topk_ind).reshape((k))

    return topk_score, topk_inds, topk_classes, topk_ys, topk_xs

def non_maximum_suppresion(hm, kernel=3, stride=1):
    pad = (kernel - 1) // 2
    hmax = [pool2d(channel, kernel, pad, stride, 'max') for channel in hm]
    keep = (hmax == hm)
    return hm * keep

def pool2d(A, kernel_size, padding=1, stride=1, pool_mode='max'):
    # padding
    A = np.pad(A, padding, mode='constant')

    # window view
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = np.lib.stride_tricks.as_strided(A, shape=output_shape + kernel_size,
                        strides=(stride*A.strides[0],
                                stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)
    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'mean':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)

def gather_feat(feat, ind):
    dim = feat.shape[1]
    ind = np.expand_dims(ind, axis=1)
    ind = np.repeat(ind, dim, axis=1)
    feat = feat[ind, np.arange(feat.shape[1])]
    return feat

def ddd_decode(heat, rot, depth, dim, wh=None, reg=None, k=40):
    cat, height, width = heat.shape
   
    # perform nms on heatmaps
    heat = non_maximum_suppresion(heat)
    
    # extract topk
    scores, inds, classes, ys, xs = topk(heat, k=k)

    # transpose and gather feat
    rot = np.transpose(rot, (1, 2, 0))
    rot = rot.reshape((-1, rot.shape[2]))
    rot = gather_feat(rot, inds)
    rot = rot.reshape((k, 8))

    depth = np.transpose(depth, (1, 2, 0))
    depth = depth.reshape((-1, depth.shape[2]))
    depth = gather_feat(depth, inds)
    depth = depth.reshape((k, 1))
    
    dim = np.transpose(dim, (1, 2, 0))
    dim = dim.reshape((-1, dim.shape[2]))
    dim = gather_feat(dim, inds)
    dim = dim.reshape((k, 3))

    wh = np.transpose(wh, (1, 2, 0))
    wh = wh.reshape((-1, wh.shape[2]))
    wh = gather_feat(wh, inds)
    wh = wh.reshape((k, 2))
    
    reg = np.transpose(reg, (1, 2, 0))
    reg = reg.reshape((-1, reg.shape[2]))
    reg = gather_feat(reg, inds)
    reg = reg.reshape((k, 2))

    
    xs = xs.reshape((k, 1)) + reg[:, 0:1]
    ys = ys.reshape((k, 1)) + reg[:, 1:2]


    classes = classes.reshape((k, 1))
    scores = scores.reshape((k, 1))
      
    detections = np.concatenate((xs, ys, scores, rot, depth, dim, wh, classes), axis=1)
    
    return detections

# ---------------------------------------------------
# Pre-processing functions
# ---------------------------------------------------

def pre_process(image, scale=1, calib=None):
    # extract image center, scale, size for post-processing
    height, width = image.shape[0:2]
    
    inp_height, inp_width = opt.input_h, opt.input_w
    c = np.array([width / 2, height / 2], dtype=np.float32)
    if opt.keep_res:
        s = np.array([inp_width, inp_height], dtype=np.int32)
    else:
        s = np.array([width, height], dtype=np.int32)

    # scale mean and std to predefined calibration values, transpose matrix to channel-first
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = image #cv2.resize(image, (width, height))
    inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
    inp_image = (inp_image.astype(np.float32) / 255.)
    inp_image = (inp_image - MEAN) / STD
    images = inp_image.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    calib = np.array(calib, dtype=np.float32) if calib is not None \
            else CALIB
    
    meta = {'c': c, 's': s, 
            'out_height': inp_height // opt.down_ratio, 
            'out_width': inp_width // opt.down_ratio,
            'calib': calib}
    return images, meta
