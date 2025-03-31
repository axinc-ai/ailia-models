
from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np
import warnings
import logging
import numpy as np
import cv2

from typing import Union
import torch


class Boxes:

    def __init__(self, tensor):

        if tensor.size == 0:
            tensor = np.reshape((-1, 4))

        self.tensor = tensor


    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert np.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"

        h, w = box_size
        x1 = np.clip(self.tensor[:, 0], 0, w)
        y1 = np.clip(self.tensor[:, 1], 0, h)
        x2 = np.clip(self.tensor[:, 2], 0, w)
        y2 = np.clip(self.tensor[:, 3], 0, h)
        self.tensor = np.stack((x1, y1, x2, y2), axis=-1)

    def nonempty(self, threshold: float = 0.0):

        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        if isinstance(item, int):
            return Boxes(self.tensor[item].reshape(1, -1))  # view → reshape
        b = self.tensor[item]
        assert b.ndim == 2, f"Indexing on Boxes with {item} failed to return a matrix!"
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]


    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0):
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside


    def scale(self, scale_x: float, scale_y: float) -> None:
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

class Instances:

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        with warnings.catch_warnings(record=True):
            data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields


    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields


    def __getitem__(self, item: Union[int, slice, Any]) -> "Instances":
        if type(item) is int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")


class BoundingBoxVisualizer:
    def __init__(self):
        self.rectangle_visualizer = RectangleVisualizer()

    def visualize(self, image_bgr, boxes_xywh):
        for bbox_xywh in boxes_xywh:
            image_bgr = self.rectangle_visualizer.visualize(image_bgr, bbox_xywh)
        return image_bgr


class ScoredBoundingBoxVisualizer:
    def __init__(self, bbox_visualizer_params=None, score_visualizer_params=None, **kwargs):
        if bbox_visualizer_params is None:
            bbox_visualizer_params = {}
        if score_visualizer_params is None:
            score_visualizer_params = {}
        self.visualizer_bbox = RectangleVisualizer(**bbox_visualizer_params)
        self.visualizer_score = TextVisualizer(**score_visualizer_params)

    def visualize(self, image_bgr, scored_bboxes):
        boxes_xywh, box_scores = scored_bboxes
        assert len(boxes_xywh) == len(
            box_scores
        ), "Number of bounding boxes {} should be equal to the number of scores {}".format(
            len(boxes_xywh), len(box_scores)
        )
        for i, box_xywh in enumerate(boxes_xywh):
            score_i = box_scores[i]
            image_bgr = self.visualizer_bbox.visualize(image_bgr, box_xywh)
            score_txt = "{0:6.4f}".format(score_i)
            topleft_xy = box_xywh[0], box_xywh[1]
            image_bgr = self.visualizer_score.visualize(image_bgr, score_txt, topleft_xy)
        return image_bgr


Image = np.ndarray


class MatrixVisualizer:
    """
    Base visualizer for matrix data
    """

    def __init__(
        self,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        val_scale=1.0,
        alpha=0.7,
        interp_method_matrix=cv2.INTER_LINEAR,
        interp_method_mask=cv2.INTER_NEAREST,
    ):
        self.inplace = inplace
        self.cmap = cmap
        self.val_scale = val_scale
        self.alpha = alpha
        self.interp_method_matrix = interp_method_matrix
        self.interp_method_mask = interp_method_mask

    def visualize(self, image_bgr, mask, matrix, bbox_xywh):
        self._check_image(image_bgr)
        self._check_mask_matrix(mask, matrix)
        if self.inplace:
            image_target_bgr = image_bgr
        else:
            image_target_bgr = image_bgr * 0
        x, y, w, h = [int(v) for v in bbox_xywh]
        if w <= 0 or h <= 0:
            return image_bgr
        mask, matrix = self._resize(mask, matrix, w, h)
        mask_bg = np.tile((mask == 0)[:, :, np.newaxis], [1, 1, 3])
        matrix_scaled = matrix.astype(np.float32) * self.val_scale
        _EPSILON = 1e-6
        if np.any(matrix_scaled > 255 + _EPSILON):
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Matrix has values > {255 + _EPSILON} after " f"scaling, clipping to [0..255]"
            )
        matrix_scaled_8u = matrix_scaled.clip(0, 255).astype(np.uint8)
        matrix_vis = cv2.applyColorMap(matrix_scaled_8u, self.cmap)
        matrix_vis[mask_bg] = image_target_bgr[y : y + h, x : x + w, :][mask_bg]
        image_target_bgr[y : y + h, x : x + w, :] = (
            image_target_bgr[y : y + h, x : x + w, :] * (1.0 - self.alpha) + matrix_vis * self.alpha
        )
        return image_target_bgr.astype(np.uint8)

class RectangleVisualizer:

    _COLOR_GREEN = (18, 127, 15)

    def __init__(self, color=_COLOR_GREEN, thickness=1):
        self.color = color
        self.thickness = thickness

    def visualize(self, image_bgr, bbox_xywh, color=None, thickness=None):
        x, y, w, h = bbox_xywh
        color = color or self.color
        thickness = thickness or self.thickness
        cv2.rectangle(image_bgr, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
        return image_bgr



class TextVisualizer:

    _COLOR_GRAY = (218, 227, 218)
    _COLOR_WHITE = (255, 255, 255)

    def __init__(
        self,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_color_bgr=_COLOR_GRAY,
        font_scale=0.35,
        font_line_type=cv2.LINE_AA,
        font_line_thickness=1,
        fill_color_bgr=_COLOR_WHITE,
        fill_color_transparency=1.0,
        frame_color_bgr=_COLOR_WHITE,
        frame_color_transparency=1.0,
        frame_thickness=1,
    ):
        self.font_face = font_face
        self.font_color_bgr = font_color_bgr
        self.font_scale = font_scale
        self.font_line_type = font_line_type
        self.font_line_thickness = font_line_thickness
        self.fill_color_bgr = fill_color_bgr
        self.fill_color_transparency = fill_color_transparency
        self.frame_color_bgr = frame_color_bgr
        self.frame_color_transparency = frame_color_transparency
        self.frame_thickness = frame_thickness

    def visualize(self, image_bgr, txt, topleft_xy):
        txt_w, txt_h = self.get_text_size_wh(txt)
        topleft_xy = tuple(map(int, topleft_xy))
        x, y = topleft_xy
        if self.frame_color_transparency < 1.0:
            t = self.frame_thickness
            image_bgr[y - t : y + txt_h + t, x - t : x + txt_w + t, :] = (
                image_bgr[y - t : y + txt_h + t, x - t : x + txt_w + t, :]
                * self.frame_color_transparency
                + np.array(self.frame_color_bgr) * (1.0 - self.frame_color_transparency)
            ).astype(float)
        if self.fill_color_transparency < 1.0:
            image_bgr[y : y + txt_h, x : x + txt_w, :] = (
                image_bgr[y : y + txt_h, x : x + txt_w, :] * self.fill_color_transparency
                + np.array(self.fill_color_bgr) * (1.0 - self.fill_color_transparency)
            ).astype(float)
        cv2.putText(
            image_bgr,
            txt,
            topleft_xy,
            self.font_face,
            self.font_scale,
            self.font_color_bgr,
            self.font_line_thickness,
            self.font_line_type,
        )
        return image_bgr

    def get_text_size_wh(self, txt):
        ((txt_w, txt_h), _) = cv2.getTextSize(
            txt, self.font_face, self.font_scale, self.font_line_thickness
        )
        return txt_w, txt_h


class CompoundVisualizer:
    def __init__(self, visualizers):
        self.visualizers = visualizers

    def visualize(self, image_bgr, data):
        assert len(data) == len(
            self.visualizers
        ), "The number of datas {} should match the number of visualizers" " {}".format(
            len(data), len(self.visualizers)
        )
        image = image_bgr
        for i, visualizer in enumerate(self.visualizers):
            image = visualizer.visualize(image, data[i])
        return image

    def __str__(self):
        visualizer_str = ", ".join([str(v) for v in self.visualizers])
        return "Compound Visualizer [{}]".format(visualizer_str)

class DensePoseEmbeddingPredictorOutput:
    """
    Predictor output that contains embedding and coarse segmentation data:
     * embedding: float tensor of size [N, D, H, W], contains estimated embeddings
     * coarse_segm: float tensor of size [N, K, H, W]
    Here D = MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_SIZE
         K = MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
    """

    embedding: torch.Tensor
    coarse_segm: torch.Tensor

    def __len__(self):
        """
        Number of instances (N) in the output
        """
        return self.coarse_segm.size(0)

    def __getitem__(
        self, item: Union[int, slice, torch.BoolTensor]
    ) -> "DensePoseEmbeddingPredictorOutput":
        """
        Get outputs for the selected instance(s)

        Args:
            item (int or slice or tensor): selected items
        """
        if isinstance(item, int):
            return DensePoseEmbeddingPredictorOutput(
                coarse_segm=self.coarse_segm[item].unsqueeze(0),
                embedding=self.embedding[item].unsqueeze(0),
            )
        else:
            return DensePoseEmbeddingPredictorOutput(
                coarse_segm=self.coarse_segm[item], embedding=self.embedding[item]
            )

    #def to(self, device: torch.device):
    #    """
    #    Transfers all tensors to the given device
    #    """
    #    coarse_segm = self.coarse_segm.to(device)
    #    embedding = self.embedding.to(device)
    #    return DensePoseEmbeddingPredictorOutput(coarse_segm=coarse_segm, embedding=embedding)

class DensePoseDataRelative:
    """
    Dense pose relative annotations that can be applied to any bounding box:
        x - normalized X coordinates [0, 255] of annotated points
        y - normalized Y coordinates [0, 255] of annotated points
        i - body part labels 0,...,24 for annotated points
        u - body part U coordinates [0, 1] for annotated points
        v - body part V coordinates [0, 1] for annotated points
        segm - 256x256 segmentation mask with values 0,...,14
    To obtain absolute x and y data wrt some bounding box one needs to first
    divide the data by 256, multiply by the respective bounding box size
    and add bounding box offset:
        x_img = x0 + x_norm * w / 256.0
        y_img = y0 + y_norm * h / 256.0
    Segmentation masks are typically sampled to get image-based masks.
    """

    # Key for normalized X coordinates in annotation dict
    X_KEY = "dp_x"
    # Key for normalized Y coordinates in annotation dict
    Y_KEY = "dp_y"
    # Key for U part coordinates in annotation dict (used in chart-based annotations)
    U_KEY = "dp_U"
    # Key for V part coordinates in annotation dict (used in chart-based annotations)
    V_KEY = "dp_V"
    # Key for I point labels in annotation dict (used in chart-based annotations)
    I_KEY = "dp_I"
    # Key for segmentation mask in annotation dict
    S_KEY = "dp_masks"
    # Key for vertex ids (used in continuous surface embeddings annotations)
    VERTEX_IDS_KEY = "dp_vertex"
    # Key for mesh id (used in continuous surface embeddings annotations)
    MESH_NAME_KEY = "ref_model"
    # Number of body parts in segmentation masks
    N_BODY_PARTS = 14
    # Number of parts in point labels
    N_PART_LABELS = 24
    MASK_SIZE = 256

