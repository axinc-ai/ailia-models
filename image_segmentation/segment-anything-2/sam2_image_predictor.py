import sys
import os
import time
from logging import getLogger

import numpy as np
import cv2

import ailia

import os
import numpy as np
import matplotlib.pyplot as plt
import ailia
from typing import Optional
from typing import Tuple

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import urlretrieve, progress_print, check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

class SAM2ImagePredictor:
    def set_image(self, image, image_encoder, onnx):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.0
        img = img - [0.485, 0.456, 0.406]
        img = img / [0.229, 0.224, 0.225]
        img = cv2.resize(img, (1024, 1024))
        img = np.expand_dims(img, 0)
        img = np.transpose(img, (0, 3, 1, 2))
        img = img.astype(np.float32)

        if onnx:
            vision_feat1, vision_feat2, vision_feat3 = image_encoder.run(None, {"input_image":img})
        else:
            vision_feat1, vision_feat2, vision_feat3 = image_encoder.run({"input_image":img})
        feats = [vision_feat1, vision_feat2, vision_feat3]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return features

    def predict(
        self,
        features,
        orig_hw,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
        prompt_encoder = None,
        mask_decoder = None,
        onnx = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Transform input prompts
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
        point_coords, point_labels, box, mask_input, normalize_coords, orig_hw
        )

        masks, iou_predictions, low_res_masks = self._predict(
            features,
            orig_hw,
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            onnx=onnx
        )

        return masks[0], iou_predictions[0], low_res_masks[0]

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, orig_hw
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            point_coords = point_coords.astype(np.float32)
            unnorm_coords = self.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=orig_hw
            )
            labels = point_labels.astype(np.int64)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = box.astype(np.float32)
            unnorm_box = self.transform_boxes(
                box, normalize=normalize_coords, orig_hw=orig_hw
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = mask_input.astype(np.float32)
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    def _predict(
        self, 
        features,
        orig_hw,
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        boxes: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        prompt_encoder = None,
        mask_decoder = None,
        onnx = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = np.ndarray([[2, 3]], dtype=np.int64)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = np.concatenate([box_coords, concat_points[0]], axis=1)
                concat_labels = np.concatenate([box_labels, concat_points[1]], axis=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        print(concat_points)


        if onnx:
            sparse_embeddings, dense_embeddings, dense_pe = prompt_encoder.run(None, {"coords":concat_points[0], "labels":concat_points[1]})
        else:
            sparse_embeddings, dense_embeddings, dense_pe = prompt_encoder.run({"coords":concat_points[0], "labels":concat_points[1]})

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level
            for feat_level in features["high_res_feats"]
        ]

        image_feature = features["image_embed"]
        if onnx:
            low_res_masks, iou_predictions, _, _  = mask_decoder.run(None, {
                "image_embeddings":image_feature,
                "image_pe": dense_pe,
                "sparse_prompt_embeddings": sparse_embeddings,
                "dense_prompt_embeddings": dense_embeddings,
                "high_res_features1":high_res_features[0],
                "high_res_features2":high_res_features[1]})
        else:
            low_res_masks, iou_predictions, _, _  = mask_decoder.run({
                "image_embeddings":image_feature,
                "image_pe": dense_pe,
                "sparse_prompt_embeddings": sparse_embeddings,
                "dense_prompt_embeddings": dense_embeddings,
                "high_res_features1":high_res_features[0],
                "high_res_features2":high_res_features[1]})

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(
            low_res_masks, orig_hw
        )
        low_res_masks = np.clip(low_res_masks, -32.0, 32.0)
        mask_threshold = 0.0
        if not return_logits:
            masks = masks > mask_threshold

        return masks, iou_predictions, low_res_masks


    def transform_coords(
        self, coords, normalize=False, orig_hw=None
    ):
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.copy()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        resolution = 1024
        coords = coords * resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes, normalize=False, orig_hw=None
    ):
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: np.ndarray, orig_hw) -> np.ndarray:
        interpolated_masks = []
        for mask in masks:
            mask = np.transpose(mask, (1, 2, 0))
            resized_mask = cv2.resize(mask, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)
            resized_mask = np.transpose(resized_mask, (2, 0, 1))
            interpolated_masks.append(resized_mask)
        interpolated_masks = np.array(interpolated_masks)

        return interpolated_masks
