import numpy as np
import cv2

# reference
# https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/box_utils.py
LABEL_PATH = './voc-model-labels.txt'


def area_of(left_top, right_bottom) -> np.array:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    # hw = torch.clamp(right_bottom - left_top, min=0.0)
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    # _, indexes = scores.sort(descending=True)
    indexes = np.argsort(-scores)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            # current_box.unsqueeze(0),
            current_box.reshape(1, current_box.shape[0])
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


# TODO if we need soft_nms, convert this to np version
# def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
#     """Soft NMS implementation.

#     References:
#         https://arxiv.org/abs/1704.04503
#         https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx

#     Args:
#         box_scores (N, 5): boxes in corner-form and probabilities.
#         score_threshold: boxes with scores less than value are not considered.
#         sigma: the parameter in score re-computation.
#             scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
#         top_k: keep top_k results. If k <= 0, keep all the results.
#     Returns:
#          picked_box_scores (K, 5): results of NMS.
#     """
#     picked_box_scores = []
#     while box_scores.size(0) > 0:
#         max_score_index = torch.argmax(box_scores[:, 4])
#         cur_box_prob = torch.tensor(box_scores[max_score_index, :])
#         picked_box_scores.append(cur_box_prob)
#         if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
#             break
#         cur_box = cur_box_prob[:-1]
#         box_scores[max_score_index, :] = box_scores[-1, :]
#         box_scores = box_scores[:-1, :]
#         ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
#         box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
#         box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
#     if len(picked_box_scores) > 0:
#         return torch.stack(picked_box_scores)
#     else:
#         return torch.tensor([])


def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    # if nms_method == "soft":
    # return soft_nms(box_scores, score_threshold, sigma, top_k)
    # else:
    return hard_nms(
        box_scores, iou_threshold, top_k, candidate_size=candidate_size
    )


def post_processing(scores, boxes, top_k=10):
    # [FUTURE WORK?] for now, boxes can take minus value
    boxes = boxes[0]
    scores = scores[0]
    prob_threshold = 0.4
    width, height = 300, 300  # fixed

    picked_box_probs = []
    picked_labels = []

    # print(f"[DEBUG] {scores.shape[1]}")
    
    for class_index in range(1, scores.shape[1]):
        probs = scores[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate(
            [subset_boxes, probs.reshape(-1, 1)], axis=1
        )
        # box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
        box_probs = nms(box_probs, nms_method=None,
                        score_threshold=prob_threshold,
                        iou_threshold=0.45,
                        sigma=0.5,
                        top_k=top_k,
                        candidate_size=200)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    # picked_box_probs = torch.cat(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return (picked_box_probs[:, :4],
            np.array(picked_labels),
            picked_box_probs[:, 4])
    

def save_result(org_image, scores, boxes):
    class_names = [name.strip() for name in open(LABEL_PATH).readlines()]
    boxes, labels, probs = post_processing(scores, boxes, top_k=10)
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(
            org_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2
        )
        
        cv2.putText(
            org_image,
            label,
            (int(box[0])+20, int(box[1])+40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # font scale
            (255, 0, 255), # font color
            1, # thickness
            cv2.LINE_AA # Line type
        )
        
    cv2.imwrite('annotated.png', org_image)
