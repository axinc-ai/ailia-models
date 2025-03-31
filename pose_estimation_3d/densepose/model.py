import numpy as np
from typing import Dict, List

from util1 import Instances

import cv2
import onnxruntime

def resize_shortest_edge(image, min_size=800, max_size=1333):
    h, w = image.shape[:2]
    min_orig = min(h, w)
    max_orig = max(h, w)

    scale = min_size / min_orig
    if round(scale * max_orig) > max_size:
        scale = max_size / max_orig

    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized_image


class DefaultPredictor:

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = resize_shortest_edge(original_image, min_size=self.cfg.INPUT.MIN_SIZE_TEST, max_size=self.cfg.INPUT.MAX_SIZE_TEST)

        image = image.astype("float32").transpose(2, 0, 1)


        session = onnxruntime.InferenceSession("test_rcnn_inference.onnx")
        input_name1 = session.get_inputs()[0].name
        results = session.run([], {input_name1: image})

        instances1,instance2, instances3,instances4,image_sizes, coarse_segm,fine_segm,u,v =  results
        image_sizes = [image_sizes]
        instances = [[[instances1,instance2,instances3]], [instances4]]

        result2 =  (coarse_segm,
                    fine_segm,
                    u,
                    v )
        return image,instances ,image_sizes, result2


class GeneralizedRCNN():

    def __init__():
        pass

    @staticmethod
    #def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
    def _postprocess(instances, batched_inputs, image_sizes):
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append(r)
        return processed_results


def detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    new_size = (output_height, output_width)
    output_width_tmp = output_width
    output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results
