import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance

from skimage.filters.rank import mean_bilateral
from skimage import morphology


def preProcess(img):
    img[:, :, 0] = mean_bilateral(
        img[:, :, 0], morphology.disk(20), s0=10, s1=10
    )
    img[:, :, 1] = mean_bilateral(
        img[:, :, 1], morphology.disk(20), s0=10, s1=10
    )
    img[:, :, 2] = mean_bilateral(
        img[:, :, 2], morphology.disk(20), s0=10, s1=10
    )
    return img


def padCropImg(img):

    H = img.shape[0]
    W = img.shape[1]

    patchRes = 128
    pH = patchRes
    pW = patchRes
    ovlp = int(patchRes * 0.125)

    padH = (int((H - patchRes)/(patchRes - ovlp) + 1)
            * (patchRes - ovlp) + patchRes) - H
    padW = (int((W - patchRes)/(patchRes - ovlp) + 1)
            * (patchRes - ovlp) + patchRes) - W

    padImg = cv2.copyMakeBorder(img, 0, padH, 0, padW, cv2.BORDER_REPLICATE)

    ynum = int((padImg.shape[0] - pH)/(pH - ovlp)) + 1
    xnum = int((padImg.shape[1] - pW)/(pW - ovlp)) + 1

    totalPatch = np.zeros((ynum, xnum, patchRes, patchRes, 3), dtype=np.uint8)

    for j in range(0, ynum):
        for i in range(0, xnum):
            x = int(i * (pW - ovlp))
            y = int(j * (pH - ovlp))
            totalPatch[j, i] = padImg[y:int(y + patchRes), x:int(x + patchRes)]

    return totalPatch


def composePatch(totalResults):
    ynum = totalResults.shape[0]
    xnum = totalResults.shape[1]
    patchRes = totalResults.shape[2]

    ovlp = int(patchRes * 0.125)
    step = patchRes - ovlp

    resImg = np.zeros((
        patchRes + (ynum - 1) * step,
        patchRes + (xnum - 1) * step,
        3
    ), np.uint8)

    for j in range(0, ynum):
        for i in range(0, xnum):
            sy = int(j*step)
            sx = int(i*step)
            resImg[sy:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i]

    return resImg


def postProcess(img):
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)
    factor = 2.0
    img = enhancer.enhance(factor)
    return img
