import numpy as np
import cv2
from skimage.measure import block_reduce

__all__ = [
    'k_resize',
    'sk_resize',
    'd_resize',
    'min_k_down',
    'mini_norm',
    'hard_norm',
    'opreate_normal_hint',
    'ini_hint',
    'de_line',
    'blur_line',
    'clip_15',
]


def k_resize(x, k):
    if x.shape[0] < x.shape[1]:
        s0 = k
        s1 = int(x.shape[1] * (k / x.shape[0]))
        s1 = s1 - s1 % 64
        _s0 = 16 * s0
        _s1 = int(x.shape[1] * (_s0 / x.shape[0]))
        _s1 = (_s1 + 32) - (_s1 + 32) % 64
    else:
        s1 = k
        s0 = int(x.shape[0] * (k / x.shape[1]))
        s0 = s0 - s0 % 64
        _s1 = 16 * s1
        _s0 = int(x.shape[0] * (_s1 / x.shape[1]))
        _s0 = (_s0 + 32) - (_s0 + 32) % 64
    new_min = min(_s1, _s0)
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4

    y = cv2.resize(x, (_s1, _s0), interpolation=interpolation)
    return y


def sk_resize(x, k):
    if x.shape[0] < x.shape[1]:
        s0 = k
        s1 = int(x.shape[1] * (k / x.shape[0]))
        s1 = s1 - s1 % 16
        _s0 = 4 * s0
        _s1 = int(x.shape[1] * (_s0 / x.shape[0]))
        _s1 = (_s1 + 8) - (_s1 + 8) % 16
    else:
        s1 = k
        s0 = int(x.shape[0] * (k / x.shape[1]))
        s0 = s0 - s0 % 16
        _s1 = 4 * s1
        _s0 = int(x.shape[0] * (_s1 / x.shape[1]))
        _s0 = (_s0 + 8) - (_s0 + 8) % 16
    new_min = min(_s1, _s0)
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (_s1, _s0), interpolation=interpolation)
    return y


def d_resize(x, d, fac=1.0):
    new_min = min(int(d[1] * fac), int(d[0] * fac))
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (int(d[1] * fac), int(d[0] * fac)), interpolation=interpolation)
    return y


def min_k_down(x, k):
    y = 255 - x.astype(np.float32)
    y = block_reduce(y, (k, k), np.max)
    y = 255 - y
    return y.clip(0, 255).astype(np.uint8)


def mini_norm(x):
    y = x.astype(np.float32)
    y = 1 - y / 255.0
    y -= np.min(y)
    y /= np.max(y)
    return (255.0 - y * 80.0).astype(np.uint8)


def hard_norm(x):
    o = x.astype(np.float32)
    b = cv2.GaussianBlur(x, (3, 3), 0).astype(np.float32)
    y = (o - b + 255.0).clip(0, 255)
    y = 1 - y / 255.0
    y -= np.min(y)
    y /= np.max(y)
    y[y < np.mean(y)] = 0
    y[y > 0] = 1
    return (255.0 - y * 255.0).astype(np.uint8)


def opreate_normal_hint(gird, points, type, length):
    h = gird.shape[0]
    w = gird.shape[1]
    for point in points:
        x, y, r, g, b, t = point
        if t == type:
            x = int(x * w)
            y = int(y * h)
            l_ = max(0, x - length)
            b_ = max(0, y - length)
            r_ = min(w, x + length + 1)
            t_ = min(h, y + length + 1)
            gird[b_:t_, l_:r_, 2] = r
            gird[b_:t_, l_:r_, 1] = g
            gird[b_:t_, l_:r_, 0] = b
            gird[b_:t_, l_:r_, 3] = 255.0
    return gird


def ini_hint(x):
    r = np.zeros(shape=(x.shape[0], x.shape[1], 4), dtype=np.float32)
    return r


def de_line(x, y):
    a = x.astype(np.float32)
    b = y.astype(np.float32)[:, :, None] / 255.0
    c = np.tile(np.array([255, 255, 255])[None, None, ::-1], [a.shape[0], a.shape[1], 1])
    return (a * b + c * (1 - b)).clip(0, 255).astype(np.uint8)


def blur_line(x, y):
    o = x.astype(np.float32)
    b = cv2.GaussianBlur(x, (3, 3), 0).astype(np.float32)
    k = y.astype(np.float32)[:, :, None] / 255.0
    return (o * k + b * (1 - k)).clip(0, 255).astype(np.uint8)


def clip_15(x, s=15.0):
    return ((x - s) / (255.0 - s - s)).clip(0, 1) * 255.0
