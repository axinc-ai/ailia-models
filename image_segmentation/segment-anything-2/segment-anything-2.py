import os
import numpy as np
import matplotlib.pyplot as plt
import ailia
from typing import Optional
from typing import Tuple

# %%
np.random.seed(3)

import torch

class PositionEmbeddingRandom():
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.positional_encoding_gaussian_matrix = scale * torch.randn((2, num_pos_feats)),

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def __call__(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid = torch.ones((h, w), dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

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

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
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
        plt.show()

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
        return_logits=return_logits
    )

    masks_np = masks.squeeze(0).float().detach().cpu().numpy()
    iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
    low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
    return masks_np, iou_predictions_np, low_res_masks_np

def _prep_prompts(
    point_coords, point_labels, box, mask_logits, normalize_coords, orig_hw
):

    unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
    if point_coords is not None:
        assert (
            point_labels is not None
        ), "point_labels must be supplied if point_coords is supplied."
        point_coords = torch.as_tensor(
            point_coords, dtype=torch.float
        )
        unnorm_coords = transform_coords(
            point_coords, normalize=normalize_coords, orig_hw=orig_hw
        )
        labels = torch.as_tensor(point_labels, dtype=torch.int)
        if len(unnorm_coords.shape) == 2:
            unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
    if box is not None:
        box = torch.as_tensor(box, dtype=torch.float,)
        unnorm_box = transform_boxes(
            box, normalize=normalize_coords, orig_hw=orig_hw
        )  # Bx2x2
    if mask_logits is not None:
        mask_input = torch.as_tensor(
            mask_logits, dtype=torch.float
        )
        if len(mask_input.shape) == 3:
            mask_input = mask_input[None, :, :, :]
    return mask_input, unnorm_coords, labels, unnorm_box

def _predict(
    features,
    orig_hw,
    point_coords: Optional[torch.Tensor],
    point_labels: Optional[torch.Tensor],
    boxes: Optional[torch.Tensor] = None,
    mask_input: Optional[torch.Tensor] = None,
    multimask_output: bool = True,
    return_logits: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if point_coords is not None:
        concat_points = (point_coords, point_labels)
    else:
        concat_points = None

    # Embed prompts
    if boxes is not None:
        box_coords = boxes.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
        box_labels = box_labels.repeat(boxes.size(0), 1)
        # we merge "boxes" and "points" into a single "concat_points" input (where
        # boxes are added at the beginning) to sam_prompt_encoder
        if concat_points is not None:
            concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
            concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
            concat_points = (concat_coords, concat_labels)
        else:
            concat_points = (box_coords, box_labels)


    model = ailia.Net(weight="prompt_encoder.onnx", stream=None, memory_mode=11, env_id=1)
    print(concat_points)
    if mask_input is None:
        mask_input = np.zeros((1, 1))
    sparse_embeddings, dense_embeddings = model.run(concat_points, mask_input)

    # Predict masks
    batched_mode = (
        concat_points is not None and concat_points[0].shape[0] > 1
    )  # multi object prediction
    high_res_features = [
        feat_level.unsqueeze(0)
        for feat_level in features["high_res_feats"]
    ]
    model = ailia.Net(weight="mask_decoder.onnx", stream=None, memory_mode=11, env_id=1)
    pe = PositionEmbeddingRandom()
    image_embedding_size = [64, 64]
    image_feature = features["image_embed"].unsqueeze(0).numpy()
    image_pe = pe(image_embedding_size).unsqueeze(0).numpy()
    low_res_masks, iou_predictions, _, _ = model.run([image_feature, image_pe,  sparse_embeddings, dense_embeddings, multimask_output, batched_mode, high_res_features])

    # Upscale the masks to the original image resolution
    masks = postprocess_masks(
        low_res_masks, orig_hw
    )
    low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
    mask_threshold = 0.0
    if not return_logits:
        masks = masks > mask_threshold

    return masks, iou_predictions, low_res_masks


def transform_coords(
    coords: torch.Tensor, normalize=False, orig_hw=None
) -> torch.Tensor:
    if normalize:
        assert orig_hw is not None
        h, w = orig_hw
        coords = coords.clone()
        coords[..., 0] = coords[..., 0] / w
        coords[..., 1] = coords[..., 1] / h

    resolution = 1024
    coords = coords * resolution  # unnormalize coords
    return coords

def transform_boxes(
    boxes: torch.Tensor, normalize=False, orig_hw=None
) -> torch.Tensor:
    boxes = transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
    return boxes

def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
    # max_hole_areaは0.0を仮定している
    # そうでない場合はオリジナルは穴埋め処理が入る
    import torch.nn.functional as F
    masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
    return masks


show = True


model = ailia.Net(weight="image_encoder.onnx", stream=None, memory_mode=11, env_id=1)

import cv2
import numpy
img = cv2.imread("truck.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(numpy.float32)
img = img / 255.0
img = img - [0.485, 0.456, 0.406]
img = img / [0.229, 0.224, 0.225]
img = cv2.resize(img, (1024, 1024))
img = numpy.expand_dims(img, 0)
img = numpy.transpose(img, (0, 3, 1, 2))
print(img.shape)
orig_hw = [img.shape[2], img.shape[3]]
feats = model.run(img)

features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

input_point = np.array([[500, 375]])
input_label = np.array([1])

masks, scores, logits = predict(
    orig_hw=orig_hw,
    features=features,
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

if show:
    show_masks(img, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

print("Success!")