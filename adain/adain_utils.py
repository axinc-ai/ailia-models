import numpy as np


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.shape
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.reshape(N, C, -1).var(axis=2) + eps
    feat_std = np.sqrt(feat_var).reshape(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(axis=2).reshape(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.shape[:2] == style_feat.shape[:2])
    size = content_feat.shape
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (
        content_feat - np.tile(content_mean, size[2:])
    ) / np.tile(content_std, size[2:])

    return normalized_feat * np.tile(style_std, size[2:]) +\
        np.tile(style_mean, size[2:])
