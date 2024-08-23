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

# ======================
# Parameters
# ======================

WEIGHT_IMAGE_ENCODER_L_PATH = 'image_encoder_hiera_l.onnx'
MODEL_IMAGE_ENCODER_L_PATH = 'image_encoder_hiera_l.onnx.prototxt'
WEIGHT_PROMPT_ENCODER_L_PATH = 'prompt_encoder_hiera_l.onnx'
MODEL_PROMPT_ENCODER_L_PATH = 'prompt_encoder_hiera_l.onnx.prototxt'
WEIGHT_MASK_DECODER_L_PATH = 'mask_decoder_hiera_l.onnx'
MODEL_MASK_DECODER_L_PATH = 'mask_decoder_hiera_l.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/segment-anything-2/'

IMAGE_PATH = 'truck.jpg'
SAVE_IMAGE_PATH = 'output.png'

POINT1 = (500, 375)
POINT2 = (1125, 625)

TARGET_LENGTH = 1024

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Segment Anything 2', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-p', '--pos', action='append', type=int, metavar="X", nargs=2,
    help='Positive coordinate specified by x,y.'
)
parser.add_argument(
    '--neg', action='append', type=int, metavar="X", nargs=2,
    help='Negative coordinate specified by x,y.'
)
parser.add_argument(
    '--box', type=int, metavar="X", nargs=4,
    help='Box coordinate specified by x1,y1,x2,y2.'
)
parser.add_argument(
    '--idx', type=int, choices=(0, 1, 2, 3),
    help='Select mask index.'
)
parser.add_argument(
    '-m', '--model_type', default='hiera_l', choices=('hiera_l'),
    help='Select model.'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)

# ======================
# Utility
# ======================

#np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, savepath = None):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        if i == 0:
            plt.savefig(savepath)

# ======================
# Logic
# ======================

def predict(
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
    mask_decoder = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Transform input prompts
    mask_input, unnorm_coords, labels, unnorm_box = _prep_prompts(
       point_coords, point_labels, box, mask_input, normalize_coords, orig_hw
    )

    masks, iou_predictions, low_res_masks = _predict(
        features,
        orig_hw,
        unnorm_coords,
        labels,
        unnorm_box,
        mask_input,
        multimask_output,
        return_logits=return_logits,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder
    )

    return masks[0], iou_predictions[0], low_res_masks[0]

def _prep_prompts(
    point_coords, point_labels, box, mask_logits, normalize_coords, orig_hw
):

    unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
    if point_coords is not None:
        point_coords = point_coords.astype(np.float32)
        unnorm_coords = transform_coords(
            point_coords, normalize=normalize_coords, orig_hw=orig_hw
        )
        labels = point_labels.astype(np.int64)
        if len(unnorm_coords.shape) == 2:
            unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
    if box is not None:
        box = box.astype(np.float32)
        unnorm_box = transform_boxes(
            box, normalize=normalize_coords, orig_hw=orig_hw
        )  # Bx2x2
    if mask_logits is not None:
        mask_input = mask_input.astype(np.float32)
        if len(mask_input.shape) == 3:
            mask_input = mask_input[None, :, :, :]
    return mask_input, unnorm_coords, labels, unnorm_box

def _predict(
    features,
    orig_hw,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    boxes: Optional[np.ndarray] = None,
    mask_input: Optional[np.ndarray] = None,
    multimask_output: bool = True,
    return_logits: bool = False,
    prompt_encoder = None,
    mask_decoder = None
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


    if args.onnx:
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
    if args.onnx:
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
    masks = postprocess_masks(
        low_res_masks, orig_hw
    )
    low_res_masks = np.clip(low_res_masks, -32.0, 32.0)
    mask_threshold = 0.0
    if not return_logits:
        masks = masks > mask_threshold

    return masks, iou_predictions, low_res_masks


def transform_coords(
    coords, normalize=False, orig_hw=None
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
    boxes, normalize=False, orig_hw=None
):
    boxes = transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
    return boxes

def postprocess_masks(masks: np.ndarray, orig_hw) -> np.ndarray:
    interpolated_masks = []
    for mask in masks:
        mask = np.transpose(mask, (1, 2, 0))
        resized_mask = cv2.resize(mask, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)
        resized_mask = np.transpose(resized_mask, (2, 0, 1))
        interpolated_masks.append(resized_mask)
    interpolated_masks = np.array(interpolated_masks)

    return interpolated_masks

# ======================
# Main
# ======================

def recognize_from_image(image_encoder, prompt_encoder, mask_decoder):
    pos_points = args.pos
    neg_points = args.neg
    box = args.box

    if pos_points is None:
        if neg_points is None and box is None:
            pos_points = [POINT1]
        else:
            pos_points = []
    if neg_points is None:
        neg_points = []
    if box is not None:
        box = np.array(box).reshape(2, 2)

    input_point = pos_points
    input_label = np.array([1])

    input_point = []
    input_label = []
    if pos_points:
        input_point.append(np.array(pos_points))
        input_label.append(np.ones(len(pos_points)))
    if neg_points:
        input_point.append(np.array(neg_points))
        input_label.append(np.zeros(len(neg_points)))

    for image_path in args.input:
        image = cv2.imread(image_path)
        orig_hw = [image.shape[0], image.shape[1]]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.0
        img = img - [0.485, 0.456, 0.406]
        img = img / [0.229, 0.224, 0.225]
        img = cv2.resize(img, (1024, 1024))
        img = np.expand_dims(img, 0)
        img = np.transpose(img, (0, 3, 1, 2))
        img = img.astype(np.float32)

        if args.onnx:
            vision_feat1, vision_feat2, vision_feat3 = image_encoder.run(None, {"input_image":img})
        else:
            vision_feat1, vision_feat2, vision_feat3 = image_encoder.run({"input_image":img})
        feats = [vision_feat1, vision_feat2, vision_feat3]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        masks, scores, logits = predict(
            orig_hw=orig_hw,
            features=features,
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True, savepath=savepath)

def main():
    # model files check and download
    check_and_download_models(WEIGHT_IMAGE_ENCODER_L_PATH, None, REMOTE_PATH)
    check_and_download_models(WEIGHT_PROMPT_ENCODER_L_PATH, None, REMOTE_PATH)
    check_and_download_models(WEIGHT_MASK_DECODER_L_PATH, None, REMOTE_PATH)

    if args.onnx:
        import onnxruntime
        image_encoder = onnxruntime.InferenceSession(WEIGHT_IMAGE_ENCODER_L_PATH)
        prompt_encoder = onnxruntime.InferenceSession(WEIGHT_PROMPT_ENCODER_L_PATH)
        mask_decoder = onnxruntime.InferenceSession(WEIGHT_MASK_DECODER_L_PATH)
    else:
        image_encoder = ailia.Net(weight=WEIGHT_IMAGE_ENCODER_L_PATH, stream=None, memory_mode=11, env_id=1)
        prompt_encoder = ailia.Net(weight=WEIGHT_PROMPT_ENCODER_L_PATH, stream=None, memory_mode=11, env_id=1)
        mask_decoder = ailia.Net(weight=WEIGHT_MASK_DECODER_L_PATH, stream=None, memory_mode=11, env_id=1)

    recognize_from_image(image_encoder, prompt_encoder, mask_decoder)

if __name__ == '__main__':
    main()
