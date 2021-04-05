from __future__ import division

import torch
import numpy as np

from .bbox import bbox_iou


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Softmax the class scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    prediction[:, :, :4] *= stride

    return prediction


# ADD SOFT NMS
def write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4, det_hm=False):
    """
        https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-4/
        prediction: (B x 10647 x 85)
        B: the number of images in a batch,
        10647: the number of bounding boxes predicted per image. (52×52+26×26+13×13)×3=10647
        85: the number of bounding box attributes. (c_x, c_y, w, h, object confidence, and 80 class scores)

        output: Num_obj × [img_index, x_1, y_1, x_2, y_2, object confidence, class_score, label_index]
    """

    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_a[:, :, :4]

    batch_size = prediction.size(0)

    output = prediction.new(1, prediction.size(2) + 1)
    write = False

    for ind in range(batch_size):
        # select the image from the batch
        image_pred = prediction[ind]

        # Get the class having maximum score, and the index of that class
        # Get rid of num_classes softmax scores
        # Add the class index and the class score of class having maximum score
        max_conf, max_conf_index = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_index = max_conf_index.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_index)
        image_pred = torch.cat(seq, 1)  # image_pred:(10647, 7) 7:[x1, y1, x2, y2, obj_score, max_conf, max_conf_index]

        # Get rid of the zero entries
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        image_pred__ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

        # filters out people id
        if det_hm:
            cls_mask = (image_pred__[:, -1] == 0).float()
            class_mask_ind = torch.nonzero(cls_mask).squeeze()
            image_pred_ = image_pred__[class_mask_ind].view(-1, 7)

            if torch.sum(cls_mask) == 0:
                return image_pred_
        else:
            image_pred_ = image_pred__

        # Get the various classes detected in the image
        try:
            img_classes = torch.unique(image_pred_[:, -1], sorted=True).float()
        except:
            continue

        # We will do NMS classwise
        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            # if nms has to be done
            if nms:
                # For each detection
                for i in range(idx):
                    # Get the IOUs of all boxes that come after the one we are looking at
                    # in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                    except ValueError:
                        break

                    except IndexError:
                        break

                    # Zero out all the detections that have IoU > threshold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i + 1:] *= iou_mask

                    #  Remove the zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            # Concatenate the batch_id of the image to the detection
            # this helps us identify which image does the detection correspond to
            # We use a linear structure to hold ALL the detections from the batch
            # the batch_dim is flattened
            # batch is identified by extra batch column

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    return output
