from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# Size of the ball bbox in pixels (fixed as we detect only ball center)
BALL_BBOX_SIZE = 40


player_threshold = 0.7 #TODO:
ball_threshold = 0.7 #TODO:


PLAYER_LABEL = 2
BALL_LABEL = 1


NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

normalize_trans = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)])


def _get_nms_kernel2d(kx: int, ky: int) -> torch.Tensor:
    """Utility function, which returns neigh2channels conv kernel"""
    numel: int = ky * kx
    center: int = numel // 2
    weight = torch.eye(numel)
    weight[center, center] = 0
    return weight.view(numel, 1, ky, kx)


class NonMaximaSuppression2d(nn.Module):
    r"""Applies non maxima suppression to filter.
    """

    def __init__(self, kernel_size: Tuple[int, int]):
        super(NonMaximaSuppression2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.padding: Tuple[int, int, int, int] = self._compute_zero_padding2d(kernel_size)
        self.kernel = _get_nms_kernel2d(*kernel_size)

    @staticmethod
    def _compute_zero_padding2d(kernel_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 2, kernel_size

        def pad(x):
            return (x - 1) // 2  # zero padding function

        ky, kx = kernel_size     # we assume a cubic kernel
        return pad(ky), pad(ky), pad(kx), pad(kx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4, x.shape
        B, CH, H, W = x.shape
        # find local maximum values

        padded = F.pad(
            torch.from_numpy(x), 
            list(self.padding)[::-1], 
            mode='replicate'
        )

        karnel = self.kernel.repeat(CH, 1, 1, 1)

        weight = karnel

        #max_non_center = F.conv2d(padded, weight, stride=1, groups=CH)
        #max_non_center = F.conv2d(padded, karnel, groups=CH)
        #max_non_center = F.conv2d(padded, torch.Tensor([18, 1, 3, 3]), stride=1, groups=CH)
    
        """
        below is from https://www.beam2d.net/blog/2021/01/31/npconv/
        """
        def conv2d(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1):
            import numpy as np

            """PyTorch-compatible simple implementation of conv2d in pure NumPy.

            NOTE: This code prioritizes simplicity, sacrificing the performance.
            It is much faster to use matmul instead of einsum (at least with NumPy 1.19.1).

            MIT License.

            """
            sY, sX = stride if isinstance(stride, (list, tuple)) else (stride, stride)
            pY, pX = padding if isinstance(padding, (list, tuple)) else (padding, padding)
            dY, dX = dilation if isinstance(dilation, (list, tuple)) else (dilation, dilation)
            N, iC, iH, iW = x.shape
            oC, iCg, kH, kW = w.shape
            pY_ex = (sY - (iH + pY * 2 - (kH - 1) * dY) % sY) % sY
            pX_ex = (sX - (iW + pX * 2 - (kW - 1) * dX) % sX) % sX
            oH = (iH + pY * 2 + pY_ex - (kH - 1) * dY) // sY
            oW = (iW + pX * 2 + pX_ex - (kW - 1) * dX) // sX

            x = np.pad(x, ((0, 0), (0, 0), (pY, pY + pY_ex), (pX, pX + pX_ex)))
            sN, sC, sH, sW = x.strides
            col = np.lib.stride_tricks.as_strided(
                x, shape=(N, groups, iCg, oH, oW, kH, kW),
                strides=(sN, sC * iCg, sC, sH * sY, sW * sX, sH * dY, sW * dX),
            )
            w = w.reshape(groups, oC // groups, iCg, kH, kW)
            y = np.einsum('ngihwkl,goikl->ngohw', col, w).reshape(N, oC, oH, oW)
            if bias is not None:
                y += bias[:, None, None]
            return y
        
        max_non_center = torch.from_numpy(conv2d(padded, weight, stride=1, groups=CH))
        max_non_center = max_non_center.view(B, CH, -1, H, W).max(dim=2)[0]

        x = torch.from_numpy(x)
        mask = x > max_non_center
        out_x = x * (mask.to(x.dtype))
        return out_x.to('cpu').detach().numpy().copy()


nms_kernel_size = (3, 3)
nms = NonMaximaSuppression2d(nms_kernel_size)


def image2tensor(image):
    # Convert PIL Image to the tensor (with normalization)
    return normalize_trans(image)


def numpy2tensor(image):
    # Convert OpenCV image to tensor (with normalization)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    return image2tensor(pil_image)


def detect_from_map(confidence_map, downscale_factor, max_detections, bbox_map=None):
    # downscale_factor: downscaling factor of the confidence map versus an original image

    # Confidence map is [B, C=2, H, W] tensor, where C=0 is background and C=1 is an object
    confidence_map = nms(confidence_map)[:, 1]
    # confidence_map is (B, H, W) tensor
    batch_size, h, w = confidence_map.shape[0], confidence_map.shape[1], confidence_map.shape[2]
    confidence_map = torch.from_numpy(confidence_map)
    confidence_map = confidence_map.view(batch_size, -1)

    values, indices = torch.sort(confidence_map, dim=-1, descending=True)
    if max_detections < indices.size(1):
        indices = indices[:, :max_detections]

    # Compute indexes of cells with detected object and convert to pixel coordinates
    xc = indices % w
    xc = xc.float() * downscale_factor + (downscale_factor - 1.) / 2.

    yc = torch.div(indices, w, rounding_mode='trunc')
    yc = yc.float() * downscale_factor + (downscale_factor - 1.) / 2.

    # Bounding boxes are encoded as a relative position of the centre (with respect to the cell centre)
    # and it's width and height in normalized coordinates (where 1 is the width/height of the player
    # feature map)
    # Position x and y of the bbox centre offset in normalized coords
    # (dx, dy, w, h)

    if bbox_map is not None:
        # bbox_map is (B, C=4, H, W) tensor
        bbox_map = torch.from_numpy(bbox_map)
        bbox_map = bbox_map.view(batch_size, 4, -1)
        # bbox_map is (B, C=4, H*W) tensor
        # Convert from relative to absolute (in pixel) values
        bbox_map[:, 0] *= w * downscale_factor
        bbox_map[:, 2] *= w * downscale_factor
        bbox_map[:, 1] *= h * downscale_factor
        bbox_map[:, 3] *= h * downscale_factor
    else:
        # For the ball bbox map is not given. Create fixed-size bboxes
        batch_size, h, w = confidence_map.size(0), confidence_map.size(-2), confidence_map.size(-1)
        bbox_map = torch.zeros((batch_size, 4, h * w), dtype=torch.float).to(confidence_map.device)
        bbox_map[:, [2, 3]] = BALL_BBOX_SIZE

    # Resultant detections (batch_size, max_detections, bbox),
    # where bbox = (x1, y1, x2, y2, confidence) in pixel coordinates
    detections = torch.zeros((batch_size, max_detections, 5), dtype=float).to(confidence_map.device)

    for n in range(batch_size):
        temp = bbox_map[n, :, indices[n]]
        # temp is (4, n_detections) tensor, with bbox details in pixel units (dx, dy, w, h)
        # where dx, dy is a displacement of the box center relative to the cell center

        # Compute bbox centers = cell center + predicted displacement
        bx = xc[n] + temp[0]
        by = yc[n] + temp[1]

        detections[n, :, 0] = bx - 0.5 * temp[2]  # x1
        detections[n, :, 2] = bx + 0.5 * temp[2]  # x2
        detections[n, :, 1] = by - 0.5 * temp[3]  # y1
        detections[n, :, 3] = by + 0.5 * temp[3]  # y2
        detections[n, :, 4] = values[n, :max_detections]

    return detections, values


def detect(player_feature_map, player_bbox, ball_feature_map):
    # Downsampling factor for ball and player feature maps
    ball_downsampling_factor = 4
    player_downsampling_factor = 16

    max_player_detections = 100
    max_ball_detections = 100


    # downscale_factor: downscaling factor of the confidence map versus an original image
    player_detections, player_values = detect_from_map(player_feature_map, player_downsampling_factor,
                                                max_player_detections, player_bbox)

    ball_detections, ball_values = detect_from_map(ball_feature_map, ball_downsampling_factor,
                                            max_ball_detections)

    # Iterate over batch elements and prepare a list with detection results
    output = []

    assert player_detections.shape[0] == ball_detections.shape[0], 'Error'

    player_det = player_detections[0, :, :]
    ball_det = ball_detections[0, :, :]

    # Filter out detections below the confidence threshold
    player_det = player_det[player_det[..., 4] >= player_threshold]
    player_boxes = player_det[..., 0:4]
    player_scores = player_det[..., 4]
    player_labels = torch.tensor([PLAYER_LABEL], dtype=torch.int64)
    player_labels = player_labels.repeat(player_det.size(0))
    ball_det = ball_det[ball_det[..., 4] >= ball_threshold]
    ball_boxes = ball_det[..., 0:4]
    ball_scores = ball_det[..., 4]
    ball_labels = torch.tensor([BALL_LABEL], dtype=torch.int64)
    ball_labels = ball_labels.repeat(ball_det.size(0))

    boxes = torch.cat([player_boxes, ball_boxes], dim=0)
    scores = torch.cat([player_scores, ball_scores], dim=0)
    labels = torch.cat([player_labels, ball_labels], dim=0)

    temp = {'boxes': boxes, 'labels': labels, 'scores': scores}
    output.append(temp)

    return output