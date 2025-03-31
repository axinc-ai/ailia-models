import sys
import cv2
import torch
import json
import numpy as np

import inspect
from typing import Any, Dict, List, Tuple, Union, Optional,Sequence
from fvcore.common.config import CfgNode as _CfgNode

from detectron2.config import CfgNode as CN

from dataclasses import make_dataclass, dataclass

from util1 import *

from densepose.converters.to_chart_result import ToChartResultConverterWithConfidences
from densepose.structures.chart import DensePoseChartPredictorOutput
#from chart.chart import DensePoseChartPredictorOutput
#from chart.to_chart_result import ToChartResultConverterWithConfidences
from chart import DensePoseChartResult, DensePoseChartResultWithConfidences


def decorate_predictor_output_class_with_confidences(BasePredictorOutput: DensePoseChartPredictorOutput) -> type:
    PredictorOutput = make_dataclass(
        BasePredictorOutput.__name__ + "WithConfidences",
        fields=[
            ("sigma_1", Optional[np.ndarray], None),
            ("sigma_2", Optional[np.ndarray], None),
            ("kappa_u", Optional[np.ndarray], None),
            ("kappa_v", Optional[np.ndarray], None),
            ("fine_segm_confidence"  , Optional[np.ndarray], None),
            ("coarse_segm_confidence", Optional[np.ndarray], None),
        ],
        bases=(BasePredictorOutput,),
    )
    # add possibility to index PredictorOutput

    def slice_if_not_none(data, item):
        if data is None:
            return None
        if isinstance(item, int):
            return data[item].unsqueeze(0)
        return data[item]

    def PredictorOutput_getitem(self, item):
        PredictorOutput = type(self)
        base_predictor_output_sliced = super(PredictorOutput, self).__getitem__(item)
        return PredictorOutput(
            **base_predictor_output_sliced.__dict__,
            coarse_segm_confidence=slice_if_not_none(self.coarse_segm_confidence, item),
            fine_segm_confidence=slice_if_not_none(self.fine_segm_confidence, item),
            sigma_1=slice_if_not_none(self.sigma_1, item),
            sigma_2=slice_if_not_none(self.sigma_2, item),
            kappa_u=slice_if_not_none(self.kappa_u, item),
            kappa_v=slice_if_not_none(self.kappa_v, item),
        )

    PredictorOutput.__getitem__ = PredictorOutput_getitem

    def PredictorOutput_to(self, device):
        """
        Transfers all tensors to the given device
        """
        PredictorOutput = type(self)
        base_predictor_output_to = super(PredictorOutput, self).to(device)  # pyre-ignore[16]

        return PredictorOutput(
            **base_predictor_output_to.__dict__,
            sigma_1=self.sigma_1,
            sigma_2=self.sigma_2,
            kappa_u=self.kappa_u,
            kappa_v=self.kappa_v,
            fine_segm_confidence  =self.fine_segm_confidence,
            coarse_segm_confidence=self.coarse_segm_confidence,
        )

    PredictorOutput.to = PredictorOutput_to

    return PredictorOutput

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

class DensePoseResultsVisualizer:
    def visualize(
        self,
        image_bgr: Image,
        results_and_boxes_xywh: Tuple[Optional[List[DensePoseChartResult]], Optional[Boxes]],
    ) -> Image:
        densepose_result, boxes_xywh = results_and_boxes_xywh
        if densepose_result is None or boxes_xywh is None:
            return image_bgr

        boxes_xywh = boxes_xywh.cpu().numpy()
        context = self.create_visualization_context(image_bgr)
        for i, result in enumerate(densepose_result):
            iuv_array = torch.cat(
                (result.labels[None].type(torch.float32), result.uv * 255.0)
            ).type(torch.uint8)
            self.visualize_iuv_arr(context, iuv_array.cpu().numpy(), boxes_xywh[i])
        image_bgr = self.context_to_image_bgr(context)
        return image_bgr

    def create_visualization_context(self, image_bgr: Image):
        return image_bgr

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh) -> None:
        pass

    def context_to_image_bgr(self, context):
        return context

    def get_image_bgr_from_context(self, context):
        return context


class DensePoseMaskedColormapResultsVisualizer(DensePoseResultsVisualizer):
    def __init__(
        self,
        data_extractor,
        segm_extractor,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        alpha=0.7,
        val_scale=1.0,
        **kwargs,
    ):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=val_scale, alpha=alpha
        )
        self.data_extractor = data_extractor
        self.segm_extractor = segm_extractor

    def context_to_image_bgr(self, context):
        return context

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh) -> None:
        image_bgr = self.get_image_bgr_from_context(context)
        matrix = self.data_extractor(iuv_arr)
        segm = self.segm_extractor(iuv_arr)
        mask = np.zeros(matrix.shape, dtype=np.uint8)
        mask[segm > 0] = 1
        image_bgr = self.mask_visualizer.visualize(image_bgr, mask, matrix, bbox_xywh)


def _extract_i_from_iuvarr(iuv_arr):
    return iuv_arr[0, :, :]


def _extract_u_from_iuvarr(iuv_arr):
    return iuv_arr[1, :, :]


def _extract_v_from_iuvarr(iuv_arr):
    return iuv_arr[2, :, :]


class DensePoseResultsMplContourVisualizer(DensePoseResultsVisualizer):
    def __init__(self, levels=10, **kwargs):
        self.levels = levels
        self.plot_args = kwargs

    def create_visualization_context(self, image_bgr: Image):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        context = {}
        context["image_bgr"] = image_bgr
        dpi = 100
        height_inches = float(image_bgr.shape[0]) / dpi
        width_inches = float(image_bgr.shape[1]) / dpi
        fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
        plt.axes([0, 0, 1, 1])
        plt.axis("off")
        context["fig"] = fig
        canvas = FigureCanvas(fig)
        context["canvas"] = canvas
        extent = (0, image_bgr.shape[1], image_bgr.shape[0], 0)
        plt.imshow(image_bgr[:, :, ::-1], extent=extent)
        return context

    def context_to_image_bgr(self, context):
        fig = context["fig"]
        w, h = map(int, fig.get_size_inches() * fig.get_dpi())
        canvas = context["canvas"]
        canvas.draw()
        image_1d = np.fromstring(canvas.tostring_rgb(), dtype="uint8")
        image_rgb = image_1d.reshape(h, w, 3)
        image_bgr = image_rgb[:, :, ::-1].copy()
        return image_bgr

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh: Boxes) -> None:
        import matplotlib.pyplot as plt

        u = _extract_u_from_iuvarr(iuv_arr).astype(float) / 255.0
        v = _extract_v_from_iuvarr(iuv_arr).astype(float) / 255.0
        extent = (
            bbox_xywh[0],
            bbox_xywh[0] + bbox_xywh[2],
            bbox_xywh[1],
            bbox_xywh[1] + bbox_xywh[3],
        )
        plt.contour(u, self.levels, extent=extent, **self.plot_args)
        plt.contour(v, self.levels, extent=extent, **self.plot_args)


class DensePoseResultsCustomContourVisualizer(DensePoseResultsVisualizer):
    """
    Contour visualization using marching squares
    """

    def __init__(self, levels=10, **kwargs):
        # TODO: colormap is hardcoded
        cmap = cv2.COLORMAP_PARULA
        if isinstance(levels, int):
            self.levels = np.linspace(0, 1, levels)
        else:
            self.levels = levels
        if "linewidths" in kwargs:
            self.linewidths = kwargs["linewidths"]
        else:
            self.linewidths = [1] * len(self.levels)
        self.plot_args = kwargs
        img_colors_bgr = cv2.applyColorMap((self.levels * 255).astype(np.uint8), cmap)
        self.level_colors_bgr = [
            [int(v) for v in img_color_bgr.ravel()] for img_color_bgr in img_colors_bgr
        ]

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh: Boxes) -> None:
        image_bgr = self.get_image_bgr_from_context(context)
        segm = _extract_i_from_iuvarr(iuv_arr)
        u = _extract_u_from_iuvarr(iuv_arr).astype(float) / 255.0
        v = _extract_v_from_iuvarr(iuv_arr).astype(float) / 255.0
        self._contours(image_bgr, u, segm, bbox_xywh)
        self._contours(image_bgr, v, segm, bbox_xywh)

    def _contours(self, image_bgr, arr, segm, bbox_xywh):
        for part_idx in range(1, DensePoseDataRelative.N_PART_LABELS + 1):
            mask = segm == part_idx
            if not np.any(mask):
                continue
            arr_min = np.amin(arr[mask])
            arr_max = np.amax(arr[mask])
            I, J = np.nonzero(mask)
            i0 = np.amin(I)
            i1 = np.amax(I) + 1
            j0 = np.amin(J)
            j1 = np.amax(J) + 1
            if (j1 == j0 + 1) or (i1 == i0 + 1):
                continue
            Nw = arr.shape[1] - 1
            Nh = arr.shape[0] - 1
            for level_idx, level in enumerate(self.levels):
                if (level < arr_min) or (level > arr_max):
                    continue
                vp = arr[i0:i1, j0:j1] >= level
                bin_codes = vp[:-1, :-1] + vp[1:, :-1] * 2 + vp[1:, 1:] * 4 + vp[:-1, 1:] * 8
                mp = mask[i0:i1, j0:j1]
                bin_mask_codes = mp[:-1, :-1] + mp[1:, :-1] * 2 + mp[1:, 1:] * 4 + mp[:-1, 1:] * 8
                it = np.nditer(bin_codes, flags=["multi_index"])
                color_bgr = self.level_colors_bgr[level_idx]
                linewidth = self.linewidths[level_idx]
                while not it.finished:
                    if (it[0] != 0) and (it[0] != 15):
                        i, j = it.multi_index
                        if bin_mask_codes[i, j] != 0:
                            self._draw_line(
                                image_bgr,
                                arr,
                                mask,
                                level,
                                color_bgr,
                                linewidth,
                                it[0],
                                it.multi_index,
                                bbox_xywh,
                                Nw,
                                Nh,
                                (i0, j0),
                            )
                    it.iternext()

    def _draw_line(
        self,
        image_bgr,
        arr,
        mask,
        v,
        color_bgr,
        linewidth,
        bin_code,
        multi_idx,
        bbox_xywh,
        Nw,
        Nh,
        offset,
    ):
        lines = self._bin_code_2_lines(arr, v, bin_code, multi_idx, Nw, Nh, offset)
        x0, y0, w, h = bbox_xywh
        x1 = x0 + w
        y1 = y0 + h
        for line in lines:
            x0r, y0r = line[0]
            x1r, y1r = line[1]
            pt0 = (int(x0 + x0r * (x1 - x0)), int(y0 + y0r * (y1 - y0)))
            pt1 = (int(x0 + x1r * (x1 - x0)), int(y0 + y1r * (y1 - y0)))
            cv2.line(image_bgr, pt0, pt1, color_bgr, linewidth)

    def _bin_code_2_lines(self, arr, v, bin_code, multi_idx, Nw, Nh, offset):
        i0, j0 = offset
        i, j = multi_idx
        i += i0
        j += j0
        v0, v1, v2, v3 = arr[i, j], arr[i + 1, j], arr[i + 1, j + 1], arr[i, j + 1]
        x0i = float(j) / Nw
        y0j = float(i) / Nh
        He = 1.0 / Nh
        We = 1.0 / Nw
        if (bin_code == 1) or (bin_code == 14):
            a = (v - v0) / (v1 - v0)
            b = (v - v0) / (v3 - v0)
            pt1 = (x0i, y0j + a * He)
            pt2 = (x0i + b * We, y0j)
            return [(pt1, pt2)]
        elif (bin_code == 2) or (bin_code == 13):
            a = (v - v0) / (v1 - v0)
            b = (v - v1) / (v2 - v1)
            pt1 = (x0i, y0j + a * He)
            pt2 = (x0i + b * We, y0j + He)
            return [(pt1, pt2)]
        elif (bin_code == 3) or (bin_code == 12):
            a = (v - v0) / (v3 - v0)
            b = (v - v1) / (v2 - v1)
            pt1 = (x0i + a * We, y0j)
            pt2 = (x0i + b * We, y0j + He)
            return [(pt1, pt2)]
        elif (bin_code == 4) or (bin_code == 11):
            a = (v - v1) / (v2 - v1)
            b = (v - v3) / (v2 - v3)
            pt1 = (x0i + a * We, y0j + He)
            pt2 = (x0i + We, y0j + b * He)
            return [(pt1, pt2)]
        elif (bin_code == 6) or (bin_code == 9):
            a = (v - v0) / (v1 - v0)
            b = (v - v3) / (v2 - v3)
            pt1 = (x0i, y0j + a * He)
            pt2 = (x0i + We, y0j + b * He)
            return [(pt1, pt2)]
        elif (bin_code == 7) or (bin_code == 8):
            a = (v - v0) / (v3 - v0)
            b = (v - v3) / (v2 - v3)
            pt1 = (x0i + a * We, y0j)
            pt2 = (x0i + We, y0j + b * He)
            return [(pt1, pt2)]
        elif bin_code == 5:
            a1 = (v - v0) / (v1 - v0)
            b1 = (v - v1) / (v2 - v1)
            pt11 = (x0i, y0j + a1 * He)
            pt12 = (x0i + b1 * We, y0j + He)
            a2 = (v - v0) / (v3 - v0)
            b2 = (v - v3) / (v2 - v3)
            pt21 = (x0i + a2 * We, y0j)
            pt22 = (x0i + We, y0j + b2 * He)
            return [(pt11, pt12), (pt21, pt22)]
        elif bin_code == 10:
            a1 = (v - v0) / (v3 - v0)
            b1 = (v - v0) / (v1 - v0)
            pt11 = (x0i + a1 * We, y0j)
            pt12 = (x0i, y0j + b1 * He)
            a2 = (v - v1) / (v2 - v1)
            b2 = (v - v3) / (v2 - v3)
            pt21 = (x0i + a2 * We, y0j + He)
            pt22 = (x0i + We, y0j + b2 * He)
            return [(pt11, pt12), (pt21, pt22)]
        return []


try:
    import matplotlib

    matplotlib.use("Agg")
    DensePoseResultsContourVisualizer = DensePoseResultsMplContourVisualizer
except ModuleNotFoundError:
    DensePoseResultsContourVisualizer = DensePoseResultsCustomContourVisualizer


class DensePoseResultsFineSegmentationVisualizer(DensePoseMaskedColormapResultsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7, **kwargs):
        super(DensePoseResultsFineSegmentationVisualizer, self).__init__(
            _extract_i_from_iuvarr,
            _extract_i_from_iuvarr,
            inplace,
            cmap,
            alpha,
            val_scale=255.0 / DensePoseDataRelative.N_PART_LABELS,
            **kwargs,
        )


class DensePoseResultsUVisualizer(DensePoseMaskedColormapResultsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7, **kwargs):
        super(DensePoseResultsUVisualizer, self).__init__(
            _extract_u_from_iuvarr,
            _extract_i_from_iuvarr,
            inplace,
            cmap,
            alpha,
            val_scale=1.0,
            **kwargs,
        )


class DensePoseResultsVVisualizer(DensePoseMaskedColormapResultsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7, **kwargs):
        super(DensePoseResultsVVisualizer, self).__init__(
            _extract_v_from_iuvarr,
            _extract_i_from_iuvarr,
            inplace,
            cmap,
            alpha,
            val_scale=1.0,
            **kwargs,
        )

def get_texture_atlas(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None

    #bgr_image = read_image(path)
    bgr_image = cv2.imread(path)
    rgb_image = np.copy(bgr_image)  # Convert BGR -> RGB
    rgb_image[:, :, :3] = rgb_image[:, :, 2::-1]  # Works with alpha channel
    return rgb_image


class DensePoseResultsVisualizerWithTexture(DensePoseResultsVisualizer):

    def __init__(self, texture_atlas, **kwargs):
        self.texture_atlas = texture_atlas
        self.body_part_size = texture_atlas.shape[0] // 6
        assert self.body_part_size == texture_atlas.shape[1] // 4

    def visualize(
        self,
        image_bgr: Image,
        results_and_boxes_xywh: Tuple[Optional[List[DensePoseChartResult]], Optional[Boxes]],
    ) -> Image:
        densepose_result, boxes_xywh = results_and_boxes_xywh
        if densepose_result is None or boxes_xywh is None:
            return image_bgr

        boxes_xywh = boxes_xywh.int().cpu().numpy()
        texture_image, alpha = self.get_texture()
        for i, result in enumerate(densepose_result):
            iuv_array = torch.cat((result.labels[None], result.uv.clamp(0, 1)))
            x, y, w, h = boxes_xywh[i]
            bbox_image = image_bgr[y : y + h, x : x + w]
            image_bgr[y : y + h, x : x + w] = self.generate_image_with_texture(
                texture_image, alpha, bbox_image, iuv_array.cpu().numpy()
            )
        return image_bgr

    def get_texture(self):
        N = self.body_part_size
        texture_image = np.zeros([24, N, N, self.texture_atlas.shape[-1]])
        for i in range(4):
            for j in range(6):
                texture_image[(6 * i + j), :, :, :] = self.texture_atlas[
                    N * j : N * (j + 1), N * i : N * (i + 1), :
                ]

        if texture_image.shape[-1] == 4:  # Image with alpha channel
            alpha = texture_image[:, :, :, -1] / 255.0
            texture_image = texture_image[:, :, :, :3]
        else:
            alpha = texture_image.sum(axis=-1) > 0

        return texture_image, alpha

    def generate_image_with_texture(self, texture_image, alpha, bbox_image_bgr, iuv_array):

        I, U, V = iuv_array
        generated_image_bgr = bbox_image_bgr.copy()

        for PartInd in range(1, 25):
            x, y = np.where(I == PartInd)
            x_index = (U[x, y] * (self.body_part_size - 1)).astype(int)
            y_index = ((1 - V[x, y]) * (self.body_part_size - 1)).astype(int)
            part_alpha = np.expand_dims(alpha[PartInd - 1, y_index, x_index], -1)
            generated_image_bgr[I == PartInd] = (
                generated_image_bgr[I == PartInd] * (1 - part_alpha)
                + texture_image[PartInd - 1, y_index, x_index] * part_alpha
            )

        return generated_image_bgr.astype(np.uint8)



Scores = Sequence[float]
DensePoseChartResultsWithConfidences = List[DensePoseChartResultWithConfidences]


def extract_scores_from_instances(instances: Instances, select=None):
    if instances.has("scores"):
        return instances.scores if select is None else instances.scores[select]
    return None


def extract_boxes_xywh_from_instances(instances: Instances, select=None):
    if instances.has("pred_boxes"):
        boxes_xywh = instances.pred_boxes.tensor.clone()
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]
        return boxes_xywh if select is None else boxes_xywh[select]
    return None


def create_extractor(visualizer: object):
    """
    Create an extractor for the provided visualizer
    """

    #<util.DensePoseResultsMplContourVisualizer object at 0x7450ecd8f280>
    #converter  <chart.to_chart_result.ToChartResultConverterWithConfidences object at 0x7450ece02ee0>

    if isinstance(visualizer, CompoundVisualizer):
        extractors = [create_extractor(v) for v in visualizer.visualizers]
        return CompoundExtractor(extractors)
    elif isinstance(visualizer, DensePoseResultsVisualizer):
        return DensePoseResultExtractor()
    elif isinstance(visualizer, ScoredBoundingBoxVisualizer):
        return CompoundExtractor([extract_boxes_xywh_from_instances, extract_scores_from_instances])
    elif isinstance(visualizer, BoundingBoxVisualizer):
        return extract_boxes_xywh_from_instances
    elif isinstance(visualizer, DensePoseOutputsVertexVisualizer):
        return DensePoseOutputsExtractor()
    else:
        return None


class BoundingBoxExtractor:
    """
    Extracts bounding boxes from instances
    """

    def __call__(self, instances: Instances):
        boxes_xywh = extract_boxes_xywh_from_instances(instances)
        return boxes_xywh


class ScoredBoundingBoxExtractor:
    """
    Extracts bounding boxes from instances
    """

    def __call__(self, instances: Instances, select=None):
        scores = extract_scores_from_instances(instances)
        boxes_xywh = extract_boxes_xywh_from_instances(instances)
        if (scores is None) or (boxes_xywh is None):
            return (boxes_xywh, scores)
        if select is not None:
            scores = scores[select]
            boxes_xywh = boxes_xywh[select]
        return (boxes_xywh, scores)


class DensePoseResultExtractor:
    """
    Extracts DensePose chart result with confidences from instances
    """

    def __call__(
        self, instances: Instances, select=None
    #) -> Tuple[Optional[DensePoseChartResultsWithConfidences], Optional[torch.Tensor]]:
    ) -> Tuple[Optional[DensePoseChartResultsWithConfidences], Optional[Any]]:
        if instances.has("pred_densepose") and instances.has("pred_boxes"):
            dpout = instances.pred_densepose
            boxes_xyxy = instances.pred_boxes
            boxes_xywh = extract_boxes_xywh_from_instances(instances)
            if select is not None:
                dpout = dpout[select]
                boxes_xyxy = boxes_xyxy[select]
            converter = ToChartResultConverterWithConfidences()
            results = [converter.convert(dpout[i], boxes_xyxy[[i]]) for i in range(len(dpout))]
            return results, boxes_xywh
        else:
            return None, None


class DensePoseOutputsExtractor:
    """
    Extracts DensePose result from instances
    """

    def __call__(
        self,
        instances: Instances,
        select=None,
    ) -> Tuple[
        #Optional[DensePoseEmbeddingPredictorOutput], Optional[torch.Tensor], Optional[List[int]]
        Optional[DensePoseEmbeddingPredictorOutput], Optional[Any], Optional[List[int]]
    ]:
        if not (instances.has("pred_densepose") and instances.has("pred_boxes")):
            return None, None, None

        dpout = instances.pred_densepose
        boxes_xyxy = instances.pred_boxes
        boxes_xywh = extract_boxes_xywh_from_instances(instances)

        if instances.has("pred_classes"):
            classes = instances.pred_classes.tolist()
        else:
            classes = None

        if select is not None:
            dpout = dpout[select]
            boxes_xyxy = boxes_xyxy[select]
            if classes is not None:
                classes = classes[select]

        return dpout, boxes_xywh, classes


class CompoundExtractor:
    """
    Extracts data for CompoundVisualizer
    """

    def __init__(self, extractors):
        self.extractors = extractors

    def __call__(self, instances: Instances, select=None):
        datas = []
        for extractor in self.extractors:
            data = extractor(instances, select)
            datas.append(data)
        return datas

class ScoreThresholdedExtractor:
    """
    Extracts data in the format accepted by ScoreThresholdedVisualizer
    """

    def __init__(self, extractor, min_score):
        self.extractor = extractor
        self.min_score = min_score

    def __call__(self, instances: Instances, select=None):
        scores = extract_scores_from_instances(instances)
        if scores is None:
            return None
        select_local = scores > self.min_score
        select = select_local if select is None else (select & select_local)
        data = self.extractor(instances, select=select)
        return data


class DensePoseOutputsVertexVisualizer:
    def __init__(
        self,
        cfg,
        inplace=True,
        cmap=cv2.COLORMAP_JET,
        alpha=0.7,
        device="cuda",
        default_class=0,
        **kwargs,
    ):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=1.0, alpha=alpha
        )
        self.class_to_mesh_name = get_class_to_mesh_name_mapping(cfg)
        self.embedder = build_densepose_embedder(cfg)
        self.device = torch.device(device)
        self.default_class = default_class

        self.mesh_vertex_embeddings = {
            mesh_name: self.embedder(mesh_name).to(self.device)
            for mesh_name in self.class_to_mesh_name.values()
            if self.embedder.has_embeddings(mesh_name)
        }

    def visualize(
        self,
        image_bgr: Image,
        outputs_boxes_xywh_classes: Tuple[
            Optional[DensePoseEmbeddingPredictorOutput], Optional[Boxes], Optional[List[int]]
        ],
    ) -> Image:
        if outputs_boxes_xywh_classes[0] is None:
            return image_bgr

        S, E, N, bboxes_xywh, pred_classes = self.extract_and_check_outputs_and_boxes(
            outputs_boxes_xywh_classes
        )

        for n in range(N):
            x, y, w, h = bboxes_xywh[n].int().tolist()
            mesh_name = self.class_to_mesh_name[pred_classes[n]]
            closest_vertices, mask = get_closest_vertices_mask_from_ES(
                E[[n]],
                S[[n]],
                h,
                w,
                self.mesh_vertex_embeddings[mesh_name],
                self.device,
            )
            embed_map = get_xyz_vertex_embedding(mesh_name, self.device)
            vis = (embed_map[closest_vertices].clip(0, 1) * 255.0).cpu().numpy()
            mask_numpy = mask.cpu().numpy().astype(dtype=np.uint8)
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask_numpy, vis, [x, y, w, h])

        return image_bgr

def get_texture_atlases(json_str: Optional[str]) -> Optional[Dict[str, Optional[np.ndarray]]]:
    """
    json_str is a JSON string representing a mesh_name -> texture_atlas_path dictionary
    """
    if json_str is None:
        return None

    paths = json.loads(json_str)
    return {mesh_name: get_texture_atlas(path) for mesh_name, path in paths.items()}


class DensePoseOutputsTextureVisualizer(DensePoseOutputsVertexVisualizer):
    def __init__(
        self,
        cfg,
        texture_atlases_dict,
        device="cuda",
        default_class=0,
        **kwargs,
    ):
        self.embedder = build_densepose_embedder(cfg)

        self.texture_image_dict = {}
        self.alpha_dict = {}

        for mesh_name in texture_atlases_dict.keys():
            if texture_atlases_dict[mesh_name].shape[-1] == 4:  # Image with alpha channel
                self.alpha_dict[mesh_name] = texture_atlases_dict[mesh_name][:, :, -1] / 255.0
                self.texture_image_dict[mesh_name] = texture_atlases_dict[mesh_name][:, :, :3]
            else:
                self.alpha_dict[mesh_name] = texture_atlases_dict[mesh_name].sum(axis=-1) > 0
                self.texture_image_dict[mesh_name] = texture_atlases_dict[mesh_name]

        self.device = torch.device(device)
        self.class_to_mesh_name = get_class_to_mesh_name_mapping(cfg)
        self.default_class = default_class

        self.mesh_vertex_embeddings = {
            mesh_name: self.embedder(mesh_name).to(self.device)
            for mesh_name in self.class_to_mesh_name.values()
        }

    def visualize(
        self,
        image_bgr: Image,
        outputs_boxes_xywh_classes: Tuple[
            Optional[DensePoseEmbeddingPredictorOutput], Optional[Boxes], Optional[List[int]]
        ],
    ) -> Image:
        image_target_bgr = image_bgr.copy()
        if outputs_boxes_xywh_classes[0] is None:
            return image_target_bgr

        S, E, N, bboxes_xywh, pred_classes = self.extract_and_check_outputs_and_boxes(
            outputs_boxes_xywh_classes
        )

        meshes = {
            p: create_mesh(self.class_to_mesh_name[p], self.device) for p in np.unique(pred_classes)
        }

        for n in range(N):
            x, y, w, h = bboxes_xywh[n].int().cpu().numpy()
            mesh_name = self.class_to_mesh_name[pred_classes[n]]
            closest_vertices, mask = get_closest_vertices_mask_from_ES(
                E[[n]],
                S[[n]],
                h,
                w,
                self.mesh_vertex_embeddings[mesh_name],
                self.device,
            )
            uv_array = meshes[pred_classes[n]].texcoords[closest_vertices].permute((2, 0, 1))
            uv_array = uv_array.cpu().numpy().clip(0, 1)
            textured_image = self.generate_image_with_texture(
                image_target_bgr[y : y + h, x : x + w],
                uv_array,
                mask.cpu().numpy(),
                self.class_to_mesh_name[pred_classes[n]],
            )
            if textured_image is None:
                continue
            image_target_bgr[y : y + h, x : x + w] = textured_image

        return image_target_bgr

def get_texture_atlas(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None

    #bgr_image = read_image(path)
    bgr_image = cv2.imread(path)
    rgb_image = np.copy(bgr_image)  # Convert BGR -> RGB
    rgb_image[:, :, :3] = rgb_image[:, :, 2::-1]  # Works with alpha channel
    return rgb_image


class DensePoseResultsVisualizerWithTexture(DensePoseResultsVisualizer):

    def __init__(self, texture_atlas, **kwargs):
        self.texture_atlas = texture_atlas
        self.body_part_size = texture_atlas.shape[0] // 6
        assert self.body_part_size == texture_atlas.shape[1] // 4

    def visualize(
        self,
        image_bgr: Image,
        results_and_boxes_xywh: Tuple[Optional[List[DensePoseChartResult]], Optional[Boxes]],
    ) -> Image:
        densepose_result, boxes_xywh = results_and_boxes_xywh
        if densepose_result is None or boxes_xywh is None:
            return image_bgr

        boxes_xywh = boxes_xywh.int().cpu().numpy()
        texture_image, alpha = self.get_texture()
        for i, result in enumerate(densepose_result):
            iuv_array = torch.cat((result.labels[None], result.uv.clamp(0, 1)))
            x, y, w, h = boxes_xywh[i]
            bbox_image = image_bgr[y : y + h, x : x + w]
            image_bgr[y : y + h, x : x + w] = self.generate_image_with_texture(
                texture_image, alpha, bbox_image, iuv_array.cpu().numpy()
            )
        return image_bgr

    def get_texture(self):
        N = self.body_part_size
        texture_image = np.zeros([24, N, N, self.texture_atlas.shape[-1]])
        for i in range(4):
            for j in range(6):
                texture_image[(6 * i + j), :, :, :] = self.texture_atlas[
                    N * j : N * (j + 1), N * i : N * (i + 1), :
                ]

        if texture_image.shape[-1] == 4:  # Image with alpha channel
            alpha = texture_image[:, :, :, -1] / 255.0
            texture_image = texture_image[:, :, :, :3]
        else:
            alpha = texture_image.sum(axis=-1) > 0

        return texture_image, alpha

    def generate_image_with_texture(self, texture_image, alpha, bbox_image_bgr, iuv_array):

        I, U, V = iuv_array
        generated_image_bgr = bbox_image_bgr.copy()

        for PartInd in range(1, 25):
            x, y = np.where(I == PartInd)
            x_index = (U[x, y] * (self.body_part_size - 1)).astype(int)
            y_index = ((1 - V[x, y]) * (self.body_part_size - 1)).astype(int)
            part_alpha = np.expand_dims(alpha[PartInd - 1, y_index, x_index], -1)
            generated_image_bgr[I == PartInd] = (
                generated_image_bgr[I == PartInd] * (1 - part_alpha)
                + texture_image[PartInd - 1, y_index, x_index] * part_alpha
            )

        return generated_image_bgr.astype(np.uint8)


def add_dataset_category_config(cfg: CN) -> None:
    """
    Add config for additional category-related dataset options
     - category whitelisting
     - category mapping
    """
    _C = cfg
    _C.DATASETS.CATEGORY_MAPS = CN(new_allowed=True)
    _C.DATASETS.WHITELISTED_CATEGORIES = CN(new_allowed=True)
    # class to mesh mapping
    _C.DATASETS.CLASS_TO_MESH_NAME_MAPPING = CN(new_allowed=True)


def add_evaluation_config(cfg: CN) -> None:
    _C = cfg
    _C.DENSEPOSE_EVALUATION = CN()
    # evaluator type, possible values:
    #  - "iou": evaluator for models that produce iou data
    #  - "cse": evaluator for models that produce cse data
    _C.DENSEPOSE_EVALUATION.TYPE = "iou"
    # storage for DensePose results, possible values:
    #  - "none": no explicit storage, all the results are stored in the
    #            dictionary with predictions, memory intensive;
    #            historically the default storage type
    #  - "ram": RAM storage, uses per-process RAM storage, which is
    #           reduced to a single process storage on later stages,
    #           less memory intensive
    #  - "file": file storage, uses per-process file-based storage,
    #            the least memory intensive, but may create bottlenecks
    #            on file system accesses
    _C.DENSEPOSE_EVALUATION.STORAGE = "none"
    # minimum threshold for IOU values: the lower its values is,
    # the more matches are produced (and the higher the AP score)
    _C.DENSEPOSE_EVALUATION.MIN_IOU_THRESHOLD = 0.5
    # Non-distributed inference is slower (at inference time) but can avoid RAM OOM
    _C.DENSEPOSE_EVALUATION.DISTRIBUTED_INFERENCE = True
    # evaluate mesh alignment based on vertex embeddings, only makes sense in CSE context
    _C.DENSEPOSE_EVALUATION.EVALUATE_MESH_ALIGNMENT = False
    # meshes to compute mesh alignment for
    _C.DENSEPOSE_EVALUATION.MESH_ALIGNMENT_MESH_NAMES = []


def add_bootstrap_config(cfg: CN) -> None:
    """ """
    _C = cfg
    _C.BOOTSTRAP_DATASETS = []
    _C.BOOTSTRAP_MODEL = CN()
    _C.BOOTSTRAP_MODEL.WEIGHTS = ""
    _C.BOOTSTRAP_MODEL.DEVICE = "cuda"


def get_bootstrap_dataset_config() -> CN:
    _C = CN()
    _C.DATASET = ""
    # ratio used to mix data loaders
    _C.RATIO = 0.1
    # image loader
    _C.IMAGE_LOADER = CN(new_allowed=True)
    _C.IMAGE_LOADER.TYPE = ""
    _C.IMAGE_LOADER.BATCH_SIZE = 4
    _C.IMAGE_LOADER.NUM_WORKERS = 4
    _C.IMAGE_LOADER.CATEGORIES = []
    _C.IMAGE_LOADER.MAX_COUNT_PER_CATEGORY = 1_000_000
    _C.IMAGE_LOADER.CATEGORY_TO_CLASS_MAPPING = CN(new_allowed=True)
    # inference
    _C.INFERENCE = CN()
    # batch size for model inputs
    _C.INFERENCE.INPUT_BATCH_SIZE = 4
    # batch size to group model outputs
    _C.INFERENCE.OUTPUT_BATCH_SIZE = 2
    # sampled data
    _C.DATA_SAMPLER = CN(new_allowed=True)
    _C.DATA_SAMPLER.TYPE = ""
    _C.DATA_SAMPLER.USE_GROUND_TRUTH_CATEGORIES = False
    # filter
    _C.FILTER = CN(new_allowed=True)
    _C.FILTER.TYPE = ""
    return _C



def add_densepose_head_cse_config(cfg: CN) -> None:
    """
    Add configuration options for Continuous Surface Embeddings (CSE)
    """
    _C = cfg
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE = CN()
    # Dimensionality D of the embedding space
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_SIZE = 16
    # Embedder specifications for various mesh IDs
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDERS = CN(new_allowed=True)
    # normalization coefficient for embedding distances
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDING_DIST_GAUSS_SIGMA = 0.01
    # normalization coefficient for geodesic distances
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.GEODESIC_DIST_GAUSS_SIGMA = 0.01
    # embedding loss weight
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_LOSS_WEIGHT = 0.6
    # embedding loss name, currently the following options are supported:
    # - EmbeddingLoss: cross-entropy on vertex labels
    # - SoftEmbeddingLoss: cross-entropy on vertex label combined with
    #    Gaussian penalty on distance between vertices
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_LOSS_NAME = "EmbeddingLoss"
    # optimizer hyperparameters
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.FEATURES_LR_FACTOR = 1.0
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDING_LR_FACTOR = 1.0
    # Shape to shape cycle consistency loss parameters:
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.SHAPE_TO_SHAPE_CYCLE_LOSS = CN({"ENABLED": False})
    # shape to shape cycle consistency loss weight
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.SHAPE_TO_SHAPE_CYCLE_LOSS.WEIGHT = 0.025
    # norm type used for loss computation
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.SHAPE_TO_SHAPE_CYCLE_LOSS.NORM_P = 2
    # normalization term for embedding similarity matrices
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.SHAPE_TO_SHAPE_CYCLE_LOSS.TEMPERATURE = 0.05
    # maximum number of vertices to include into shape to shape cycle loss
    # if negative or zero, all vertices are considered
    # if positive, random subset of vertices of given size is considered
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.SHAPE_TO_SHAPE_CYCLE_LOSS.MAX_NUM_VERTICES = 4936
    # Pixel to shape cycle consistency loss parameters:
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS = CN({"ENABLED": False})
    # pixel to shape cycle consistency loss weight
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.WEIGHT = 0.0001
    # norm type used for loss computation
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.NORM_P = 2
    # map images to all meshes and back (if false, use only gt meshes from the batch)
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.USE_ALL_MESHES_NOT_GT_ONLY = False
    # Randomly select at most this number of pixels from every instance
    # if negative or zero, all vertices are considered
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.NUM_PIXELS_TO_SAMPLE = 100
    # normalization factor for pixel to pixel distances (higher value = smoother distribution)
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.PIXEL_SIGMA = 5.0
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.TEMPERATURE_PIXEL_TO_VERTEX = 0.05
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.TEMPERATURE_VERTEX_TO_PIXEL = 0.05


def add_densepose_head_config(cfg: CN) -> None:
    """
    Add config for densepose head.
    """
    _C = cfg

    _C.MODEL.DENSEPOSE_ON = True

    _C.MODEL.ROI_DENSEPOSE_HEAD = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.NAME = ""
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS = 8
    # Number of parts used for point labels
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES = 24
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL = 4
    _C.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM = 512
    _C.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL = 3
    _C.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE = 2
    _C.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE = 112
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE = "ROIAlignV2"
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION = 28
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO = 2
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS = 2  # 15 or 2
    # Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
    _C.MODEL.ROI_DENSEPOSE_HEAD.FG_IOU_THRESHOLD = 0.7
    # Loss weights for annotation masks.(14 Parts)
    _C.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS = 5.0
    # Loss weights for surface parts. (24 Parts)
    _C.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS = 1.0
    # Loss weights for UV regression.
    _C.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS = 0.01
    # Coarse segmentation is trained using instance segmentation task data
    _C.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS = False
    # For Decoder
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON = True
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NUM_CLASSES = 256
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS = 256
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NORM = ""
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE = 4
    # For DeepLab head
    _C.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM = "GN"
    _C.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NONLOCAL_ON = 0
    # Predictor class name, must be registered in DENSEPOSE_PREDICTOR_REGISTRY
    # Some registered predictors:
    #   "DensePoseChartPredictor": predicts segmentation and UV coordinates for predefined charts
    #   "DensePoseChartWithConfidencePredictor": predicts segmentation, UV coordinates
    #       and associated confidences for predefined charts (default)
    #   "DensePoseEmbeddingWithConfidencePredictor": predicts segmentation, embeddings
    #       and associated confidences for CSE
    _C.MODEL.ROI_DENSEPOSE_HEAD.PREDICTOR_NAME = "DensePoseChartWithConfidencePredictor"
    # Loss class name, must be registered in DENSEPOSE_LOSS_REGISTRY
    # Some registered losses:
    #   "DensePoseChartLoss": loss for chart-based models that estimate
    #      segmentation and UV coordinates
    #   "DensePoseChartWithConfidenceLoss": loss for chart-based models that estimate
    #      segmentation, UV coordinates and the corresponding confidences (default)
    _C.MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME = "DensePoseChartWithConfidenceLoss"
    # Confidences
    # Enable learning UV confidences (variances) along with the actual values
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE = CN({"ENABLED": False})
    # UV confidence lower bound
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE.EPSILON = 0.01
    # Enable learning segmentation confidences (variances) along with the actual values
    _C.MODEL.ROI_DENSEPOSE_HEAD.SEGM_CONFIDENCE = CN({"ENABLED": False})
    # Segmentation confidence lower bound
    _C.MODEL.ROI_DENSEPOSE_HEAD.SEGM_CONFIDENCE.EPSILON = 0.01
    # Statistical model type for confidence learning, possible values:
    # - "iid_iso": statistically independent identically distributed residuals
    #    with isotropic covariance
    # - "indep_aniso": statistically independent residuals with anisotropic
    #    covariances
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE.TYPE = "iid_iso"
    # List of angles for rotation in data augmentation during training
    _C.INPUT.ROTATION_ANGLES = [0]
    _C.TEST.AUG.ROTATION_ANGLES = ()  # Rotation TTA

    add_densepose_head_cse_config(cfg)


def add_hrnet_config(cfg: CN) -> None:
    """
    Add config for HRNet backbone.
    """
    _C = cfg

    # For HigherHRNet w32
    _C.MODEL.HRNET = CN()
    _C.MODEL.HRNET.STEM_INPLANES = 64
    _C.MODEL.HRNET.STAGE2 = CN()
    _C.MODEL.HRNET.STAGE2.NUM_MODULES = 1
    _C.MODEL.HRNET.STAGE2.NUM_BRANCHES = 2
    _C.MODEL.HRNET.STAGE2.BLOCK = "BASIC"
    _C.MODEL.HRNET.STAGE2.NUM_BLOCKS = [4, 4]
    _C.MODEL.HRNET.STAGE2.NUM_CHANNELS = [32, 64]
    _C.MODEL.HRNET.STAGE2.FUSE_METHOD = "SUM"
    _C.MODEL.HRNET.STAGE3 = CN()
    _C.MODEL.HRNET.STAGE3.NUM_MODULES = 4
    _C.MODEL.HRNET.STAGE3.NUM_BRANCHES = 3
    _C.MODEL.HRNET.STAGE3.BLOCK = "BASIC"
    _C.MODEL.HRNET.STAGE3.NUM_BLOCKS = [4, 4, 4]
    _C.MODEL.HRNET.STAGE3.NUM_CHANNELS = [32, 64, 128]
    _C.MODEL.HRNET.STAGE3.FUSE_METHOD = "SUM"
    _C.MODEL.HRNET.STAGE4 = CN()
    _C.MODEL.HRNET.STAGE4.NUM_MODULES = 3
    _C.MODEL.HRNET.STAGE4.NUM_BRANCHES = 4
    _C.MODEL.HRNET.STAGE4.BLOCK = "BASIC"
    _C.MODEL.HRNET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
    _C.MODEL.HRNET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
    _C.MODEL.HRNET.STAGE4.FUSE_METHOD = "SUM"

    _C.MODEL.HRNET.HRFPN = CN()
    _C.MODEL.HRNET.HRFPN.OUT_CHANNELS = 256


def add_densepose_config(cfg: CN) -> None:
    add_densepose_head_config(cfg)
    add_hrnet_config(cfg)
    add_bootstrap_config(cfg)
    add_dataset_category_config(cfg)
    add_evaluation_config(cfg)



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



