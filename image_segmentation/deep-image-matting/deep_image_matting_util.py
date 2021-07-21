# Generate trimap using segmentation

import os
import sys
import time

import numpy as np
import cv2

# ======================
# Segmentation util
# ======================

def norm(pred):
    ma = np.max(pred)
    mi = np.min(pred)
    return (pred - mi) / (ma - mi)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def erode_and_dilate(mask, k_size, ite):
    kernel = np.ones(k_size, np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=ite)
    dilated = cv2.dilate(mask, kernel, iterations=ite)
    trimap = np.full(mask.shape, 128)
    trimap[eroded >= 254] = 255
    trimap[dilated <= 1] = 0
    return trimap


def deeplabv3_preprocess(input_data):
    input_data = input_data / 127.5 - 1.0
    input_data = input_data.transpose((2, 0, 1))[np.newaxis, :, :, :]
    return input_data


def imagenet_preprocess(input_data):
    input_data = input_data / 255.0
    input_data[:, :, 0] = (input_data[:, :, 0]-0.485)/0.229
    input_data[:, :, 1] = (input_data[:, :, 1]-0.456)/0.224
    input_data[:, :, 2] = (input_data[:, :, 2]-0.406)/0.225
    input_data = input_data.transpose((2, 0, 1))[np.newaxis, :, :, :]
    return input_data


def generate_trimap(net, input_data, args):
    src_data = input_data.copy()
    
    h = input_data.shape[0]
    w = input_data.shape[1]

    input_shape = net.get_input_shape()
    input_data = cv2.resize(input_data, (input_shape[2], input_shape[3]))

    if args.arch == "deeplabv3":
        input_data = deeplabv3_preprocess(input_data)
    if args.arch == "u2net" or args.arch == "pspnet":
        input_data = imagenet_preprocess(input_data)

    preds_ailia = net.predict([input_data])

    if args.arch == "deeplabv3":
        pred = preds_ailia[0]
        pred = pred[0, 15, :, :] / 21.0
    if args.arch == "u2net" or args.arch == "pspnet":
        pred = preds_ailia[0][0, 0, :, :]

    if args.debug:
        dump_segmentation(pred, src_data, w, h, args)

    if args.arch == "u2net":
        pred = norm(pred)
    if args.arch == "pspnet":
        pred = sigmoid(pred)

    trimap_data = cv2.resize(pred * 255, (w, h))
    trimap_data = trimap_data.reshape((h, w, 1))

    seg_data = trimap_data.copy()

    thre = 0.6
    ite = 3

    if args.arch == "deeplabv3":
        ite = 7
    if args.arch == "u2net":
        thre = 0.8
        ite = 5

    thre = 255 * thre
    trimap_data[trimap_data < thre] = 0
    trimap_data[trimap_data >= thre] = 255

    if args.arch == "deeplabv3":
        seg_data = trimap_data.copy()

    trimap_data = trimap_data.astype("uint8")

    if args.debug:
        dump_segmentation_threshold(trimap_data, src_data, w, h, args)

    trimap_data = erode_and_dilate(trimap_data, k_size=(7, 7), ite=ite)

    if args.debug:
        dump_trimap(trimap_data, src_data, w, h, args)

    return trimap_data, seg_data


# ======================
# Debug functions
# ======================

def dump_segmentation(pred, src_data, w, h, args):
    savedir = os.path.dirname(args.savepath)
    segmentation_data = cv2.resize(pred * 255, (w, h))
    segmentation_data = cv2.cvtColor(segmentation_data, cv2.COLOR_GRAY2BGR)
    segmentation_data = (src_data + segmentation_data)/2
    cv2.imwrite(os.path.join(savedir, "debug_segmentation.png"), segmentation_data)


def dump_segmentation_threshold(trimap_data, src_data, w, h, args):
    savedir = os.path.dirname(args.savepath)
    segmentation_data = trimap_data.copy()
    segmentation_data = cv2.cvtColor(segmentation_data, cv2.COLOR_GRAY2BGR)
    segmentation_data = segmentation_data.astype(np.float)
    segmentation_data = (src_data + segmentation_data)/2
    cv2.imwrite(
        os.path.join(savedir, "debug_segmentation_threshold.png"),
        segmentation_data,
    )


def dump_trimap(trimap_data, src_data, w, h, args):
    savedir = os.path.dirname(args.savepath)

    cv2.imwrite(
        os.path.join(savedir, "debug_trimap_gray.png"),
        trimap_data,
    )

    segmentation_data = trimap_data.copy().astype(np.uint8)
    segmentation_data = cv2.cvtColor(segmentation_data, cv2.COLOR_GRAY2BGR)
    segmentation_data = segmentation_data.astype(np.float)
    segmentation_data = (src_data + segmentation_data)/2

    cv2.imwrite(
        os.path.join(savedir, "debug_trimap.png"),
        segmentation_data,
    )
