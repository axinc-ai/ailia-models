import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from lib_nanodet.util.misc import multi_apply
from lib_nanodet.util.box_transform import distance2bbox
from lib_nanodet.module.nms import multiclass_nms
from lib_nanodet.transform.warp import warp_boxes


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.true_divide(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class Integral(nn.Module):
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
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

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
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


class GFLHead(nn.Module):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    :param num_classes: Number of categories excluding the background category.
    :param loss: Config of all loss functions.
    :param input_channel: Number of channels in the input feature map.
    :param feat_channels: Number of conv layers in cls and reg tower. Default: 4.
    :param stacked_convs: Number of conv layers in cls and reg tower. Default: 4.
    :param octave_base_scale: Scale factor of grid cells.
    :param strides: Down sample strides of all level feature map
    :param conv_cfg: Dictionary to construct and config conv layer. Default: None.
    :param norm_cfg: Dictionary to construct and config norm layer.
    :param reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                    in QFL setting. Default: 16.
    :param kwargs:
    """
    def __init__(self,
                 num_classes,
                 input_channel,
                 feat_channels=256,
                 stacked_convs=4,
                 octave_base_scale=4,
                 strides=[8, 16, 32],
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 reg_max=16,
                 **kwargs):
        super(GFLHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.grid_cell_scale = octave_base_scale
        self.strides = strides
        self.reg_max = reg_max

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid = True
        if self.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.distribution_project = Integral(self.reg_max)


    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.gfl_cls(cls_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred


    def post_process(self, preds, meta):
        cls_scores, bbox_preds = preds
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        preds = {}
        warp_matrix = meta['warp_matrix'][0] if isinstance(meta['warp_matrix'], list) else meta['warp_matrix']
        img_height = meta['img_info']['height'].cpu().numpy() \
            if isinstance(meta['img_info']['height'], torch.Tensor) else meta['img_info']['height']
        img_width = meta['img_info']['width'].cpu().numpy() \
            if isinstance(meta['img_info']['width'], torch.Tensor) else meta['img_info']['width']
        for result in result_list:
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height)
            classes = det_labels.cpu().numpy()
            for i in range(self.num_classes):
                inds = (classes == i)
                preds[i] = np.concatenate([
                    det_bboxes[inds, :4].astype(np.float32),
                    det_bboxes[inds, 4:5].astype(np.float32)], axis=1).tolist()
        return preds


    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   rescale=False):

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)#clsアウトプット数（３つ）
        device = cls_scores[0].device

        input_height, input_width = img_metas['img'].shape[2:]
        input_shape = [input_height, input_width]

        result_list = []
        for img_id in range(cls_scores[0].shape[0]):#image_batch数
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            scale_factor = 1
            dets = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                          input_shape, scale_factor,
                                          device, rescale)


            result_list.append(dets)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          img_shape,
                          scale_factor,
                          device,
                          rescale=False):
        """
        Decode output tensors to bboxes on one image.
        :param cls_scores: classification prediction tensors of all stages
        :param bbox_preds: regression prediction tensors of all stages
        :param img_shape: shape of input image
        :param scale_factor: scale factor of boxes
        :param device: device of the tensor
        :return: predict boxes and labels
        """
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        for stride, cls_score, bbox_pred in zip(
                self.strides, cls_scores, bbox_preds):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            featmap_size = cls_score.size()[-2:]
            y, x = self.get_single_level_center_point(
                featmap_size, stride, cls_score.dtype, device, flatten=True)
            center_points = torch.stack([x, y], dim=-1)
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.distribution_project(bbox_pred) * stride

            nms_pre = 1000
            if scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                center_points = center_points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(center_points, bbox_pred,
                                   max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        # add a dummy background class at the end of all labels, same with mmdetection2.0
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            score_thr=0.05,
            nms_cfg=dict(type='nms', iou_threshold=0.6),
            max_num=100)


        return det_bboxes, det_labels

    def get_single_level_center_point(self, featmap_size, stride, dtype, device='cuda', flatten=True):
        """
        Generate pixel centers of a single stage feature map.
        :param featmap_size: height and width of the feature map
        :param stride: down sample stride of the feature map
        :param dtype: data type of the tensors
        :param device: device of the tensors
        :param flatten: flatten the x and y tensors
        :return: y and x of the center points
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device) + 0.5) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device) + 0.5) * stride
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

    def get_grid_cells(self, featmap_size, scale, stride, dtype, device='cuda'):
        """
        Generate grid cells of a feature map for target assignment.
        :param featmap_size: Size of a single level feature map.
        :param scale: Grid cell scale.
        :param stride: Down sample stride of the feature map.
        :param dtype: Data type of the tensors.
        :param device: Device of the tensors.
        :return: Grid_cells xyxy position. Size should be [feat_w * feat_h, 4]
        """
        cell_size = stride * scale
        y, x = self.get_single_level_center_point(
            featmap_size, stride, dtype, device, flatten=True)
        grid_cells = torch.stack(
            [x - 0.5 * cell_size, y - 0.5 * cell_size,
             x + 0.5 * cell_size, y + 0.5 * cell_size], dim=-1
        )
        return grid_cells

    def grid_cells_to_center(self, grid_cells):
        """
        Get center location of each gird cell
        :param grid_cells: grid cells of a feature map
        :return: center points
        """
        cells_cx = (grid_cells[:, 2] + grid_cells[:, 0]) / 2
        cells_cy = (grid_cells[:, 3] + grid_cells[:, 1]) / 2
        return torch.stack([cells_cx, cells_cy], dim=-1)

