# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe

from dataclasses import dataclass
#import numpy as np
import torch

from typing import Any, Optional, Tuple, Union


@dataclass
class DensePoseChartPredictorOutput:
    """
    Predictor output that contains segmentation and inner coordinates predictions for predefined
    body parts:
     * coarse segmentation, a tensor of shape [N, K, Hout, Wout]
     * fine segmentation, a tensor of shape [N, C, Hout, Wout]
     * U coordinates, a tensor of shape [N, C, Hout, Wout]
     * V coordinates, a tensor of shape [N, C, Hout, Wout]
    where
     - N is the number of instances
     - K is the number of coarse segmentation channels (
         2 = foreground / background,
         15 = one of 14 body parts / background)
     - C is the number of fine segmentation channels (
         24 fine body parts / background)
     - Hout and Wout are height and width of predictions
    """

     
    #coarse_segm: np.ndarray
    #fine_segm  : np.ndarray
    #u          : np.ndarray
    #v          : np.ndarray
    coarse_segm: torch.Tensor
    fine_segm: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor

    def __len__(self):
        """
        Number of instances (N) in the output
        """
        return self.coarse_segm.size(0)
        #return self.coarse_segm.shape[0]

    def __getitem__(
        self, item: Union[int, slice, torch.BoolTensor]
    ):
        """
        Get outputs for the selected instance(s)

        Args:
            item (int or slice or tensor): selected items
        """
        if isinstance(item, int):
            return DensePoseChartPredictorOutput(
                coarse_segm=self.coarse_segm[item].unsqueeze(0),
                fine_segm  =self.fine_segm[item]  .unsqueeze(0),
                u          =self.u[item]          .unsqueeze(0),
                v          =self.v[item]          .unsqueeze(0),
            )
            #return DensePoseChartPredictorOutput(
            #    coarse_segm=np.expand_dims(self.coarse_segm[item].unsqueeze(0),
            #    fine_segm  =np.expand_dims(self.fine_segm[item]  .unsqueeze(0),
            #    u          =np.expand_dims(self.u[item]          .unsqueeze(0),
            #    v          =np.expand_dims(self.v[item]          .unsqueeze(0),
            #)
        else:
            return DensePoseChartPredictorOutput(
                coarse_segm=self.coarse_segm[item],
                fine_segm=self.fine_segm[item],
                u=self.u[item],
                v=self.v[item],
            )

#@dataclass
class DensePoseChartResult:
    """
    DensePose results for chart-based methods represented by labels and inner
    coordinates (U, V) of individual charts. Each chart is a 2D manifold
    that has an associated label and is parameterized by two coordinates U and V.
    Both U and V take values in [0, 1].
    Thus the results are represented by two tensors:
    - labels (tensor [H, W] of long): contains estimated label for each pixel of
        the detection bounding box of size (H, W)
    - uv (tensor [2, H, W] of float): contains estimated U and V coordinates
        for each pixel of the detection bounding box of size (H, W)
    """

    labels: torch.Tensor
    uv: torch.Tensor


#@dataclass
class DensePoseChartResultWithConfidences:
    """
    We add confidence values to DensePoseChartResult
    Thus the results are represented by two tensors:
    - labels (tensor [H, W] of long): contains estimated label for each pixel of
        the detection bounding box of size (H, W)
    - uv (tensor [2, H, W] of float): contains estimated U and V coordinates
        for each pixel of the detection bounding box of size (H, W)
    Plus one [H, W] tensor of float for each confidence type
    """

    labels: torch.Tensor
    uv: torch.Tensor
    sigma_1: Optional[torch.Tensor] = None
    sigma_2: Optional[torch.Tensor] = None
    kappa_u: Optional[torch.Tensor] = None
    kappa_v: Optional[torch.Tensor] = None
    fine_segm_confidence: Optional[torch.Tensor] = None
    coarse_segm_confidence: Optional[torch.Tensor] = None
