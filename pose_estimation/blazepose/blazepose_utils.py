import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

num_classes = 1
num_anchors = 896
num_coords = 12
score_clipping_thresh = 100.0
x_scale = 128.0
y_scale = 128.0
h_scale = 128.0
w_scale = 128.0
min_score_thresh = 0.75
min_suppression_threshold = 0.3
num_keypoints = 4

def resize_pad(img):
    """ resize and pad images to be input to the detectors

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio.

    Returns:
        img1: 256x256
        img2: 128x128
        scale: scale factor between original image and 256x256 image
        pad: pixels of padding in the original image
    """

    size0 = img.shape
    if size0[0]>=size0[1]:
        h1 = 256
        w1 = 256 * size0[1] // size0[0]
        padh = 0
        padw = 256 - w1
        scale = size0[1] / w1
    else:
        h1 = 256 * size0[0] // size0[1]
        w1 = 256
        padh = 256 - h1
        padw = 0
        scale = size0[0] / h1
    padh1 = padh//2
    padh2 = padh//2 + padh%2
    padw1 = padw//2
    padw2 = padw//2 + padw%2
    img1 = cv2.resize(img, (w1,h1))
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)))
    pad = (int(padh1 * scale), int(padw1 * scale))
    img2 = cv2.resize(img1, (128,128))
    return img1, img2, scale, pad

def decode_boxes(raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = torch.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(num_keypoints):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

def tensors_to_detections(raw_box_tensor, raw_score_tensor, anchors):
        """The output of the neural network is a tensor of shape (b, 896, 16)
        containing the bounding box regressor predictions, as well as a tensor 
        of shape (b, 896, 1) with the classification confidences.

        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.

        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """        
        detection_boxes = decode_boxes(raw_box_tensor, anchors)
        
        thresh = score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
        
        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))

        return output_detections

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)

def weighted_non_max_suppression(detections):
    """The alternative NMS method as mentioned in the BlazeFace paper:

    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."

    The original MediaPipe code assigns the score of the most confident
    detection to the weighted detection, but we take the average score
    of the overlapping detections.

    The input detections should be a Tensor of shape (count, 17).

    Returns a list of PyTorch tensors, one for each detected face.
    
    This is based on the source code from:
    mediapipe/calculators/util/non_max_suppression_calculator.cc
    mediapipe/calculators/util/non_max_suppression_calculator.proto
    """
    if len(detections) == 0: return []

    output_detections = []

    # Sort the detections from highest to lowest score.
    remaining = torch.argsort(detections[:, num_coords], descending=True)

    while len(remaining) > 0:
        detection = detections[remaining[0]]

        # Compute the overlap between the first box and the other 
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.)
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]
        ious = overlap_similarity(first_box, other_boxes)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        mask = ious > min_suppression_threshold
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = detection.clone()
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :num_coords]
            scores = detections[overlapping, num_coords:num_coords+1]
            total_score = scores.sum()
            weighted = (coordinates * scores).sum(dim=0) / total_score
            weighted_detection[:num_coords] = weighted
            weighted_detection[num_coords] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections  

def denormalize_detections(detections, scale, pad):
    """ maps detection coordinates from [0,1] to image coordinates

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio. This function maps the
    normalized coordinates back to the original image coordinates.

    Inputs:
        detections: nxm tensor. n is the number of detections.
            m is 4+2*k where the first 4 valuse are the bounding
            box coordinates and k is the number of additional
            keypoints output by the detector.
        scale: scalar that was used to resize the image
        pad: padding in the x and y dimensions

    """
    print(scale, pad)
    detections[:, 0] = detections[:, 0] * scale * 256 - pad[0]
    detections[:, 1] = detections[:, 1] * scale * 256 - pad[1]
    detections[:, 2] = detections[:, 2] * scale * 256 - pad[0]
    detections[:, 3] = detections[:, 3] * scale * 256 - pad[1]

    detections[:, 4::2] = detections[:, 4::2] * scale * 256 - pad[1]
    detections[:, 5::2] = detections[:, 5::2] * scale * 256 - pad[0]
    return detections

def detector_postprocess(preds, scale, pad):
    """
    docstring
    """
    anchors = torch.tensor(np.load("anchors.npy"))

    # 3. Postprocess the raw predictions:
    detections = tensors_to_detections(torch.tensor(preds[0]), torch.tensor(preds[1]), anchors)
    print(detections)

    # 4. Non-maximum suppression to remove overlapping detections:
    filtered_detections = []
    for i in range(len(detections)):
        faces = weighted_non_max_suppression(detections[i])
        faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, num_coords+1))
        filtered_detections.append(faces)

    return filtered_detections

    # print(filtered_detections)
    # pose_detections = denormalize_detections(filtered_detections[0], scale, pad)

    return pose_detections

def show_result(input_img, detections):
    for detection in detections:
        for d in detection:
            w = input_img.shape[1]
            h = input_img.shape[0]
            top_left = (int(d[1]*w), int(d[0]*h))
            bottom_right = (int(d[3]*w), int(d[2]*h))
            color = (255, 255, 255)
            cv2.rectangle(input_img, top_left, bottom_right, color, 4)

            for k in range(6):
                kp_x = d[4 + k*2] * input_img.shape[1]
                kp_y = d[4 + k*2 + 1] * input_img.shape[0]
                r = int(input_img.shape[1]/100)
                cv2.circle(input_img, (int(kp_x), int(kp_y)),
                           r, (255, 255, 255), -1)

def plot_detections(
        img, detections, with_keypoints=True, save_image_path='result.png'
):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)

    # if detections.ndim == 1:
    # detections = np.expand_dims(detections, axis=0)

    print(f'Found {detections.shape[0]} faces')
        
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]
        print(detections[i, 12].item())
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1,
            edgecolor="r",
            facecolor="none", 
            alpha=detections[i, 12].item()
        )
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(4):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = patches.Circle(
                    (kp_x, kp_y),
                    radius=0.5,
                    linewidth=1, 
                    edgecolor="lightskyblue",
                    facecolor="none", 
                    alpha=detections[i, 12].item()
                )
                ax.add_patch(circle)
    plt.savefig(save_image_path)