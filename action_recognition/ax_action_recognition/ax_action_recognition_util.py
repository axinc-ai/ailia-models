import ailia
import math
import numpy as np
import cv2

TIME_RANGE = 15

def pose_postprocess(pose_keypoints):
    thre = 0.2
    pose_keypoints[:, :, 0:2] = pose_keypoints[:, :, 0:2] - 0.5
    pose_keypoints[:, :, 0][pose_keypoints[:, :, 2] < thre] = 0
    pose_keypoints[:, :, 1][pose_keypoints[:, :, 2] < thre] = 0
    pose_keypoints[:, :, 2][pose_keypoints[:, :, 2] < thre] = 0
    return pose_keypoints

def bounding_box(person):
    threshold = 0.3
    x_min=1920
    y_min=1080
    x_max=0
    y_max=0
    for i in range(0,ailia.POSE_KEYPOINT_CNT):
        if person.points[i].score<=threshold:
            continue
        x_min=(min(x_min,person.points[i].x))
        y_min=(min(y_min,person.points[i].y))
        x_max=(max(x_max,person.points[i].x))
        y_max=(max(y_max,person.points[i].y))
    w = x_max - x_min
    h = y_max - y_min

    margin = 1.1
    px = (w * margin - w)
    py = (h * margin - h)

    x_min = x_min-px
    y_min = y_min-py
    x_max = x_max+px
    y_max = y_max+py

    x_min = min(1,max(0,x_min))
    y_min = min(1,max(0,y_min))
    x_max = min(1,max(0,x_max))
    y_max = min(1,max(0,y_max))

    return x_min,y_min,x_max-x_min,y_max-y_min

def get_detector_result_lw_human_pose(pose, h, w, get_all=False):
    xywh = []
    cls_conf = []
    cls_ids = []

    for idx in range(pose.get_object_count()):
        obj = pose.get_object_pose(idx)

        obj_x,obj_y,obj_w,obj_h = bounding_box(obj)
        prob=1
        cls_id=0

        w2=obj_w * w
        h2=obj_h * h

        #w2=max(1,w2)
        #h2=max(1,h2)

        if (w2<1 or h2<1) and (not get_all):
            continue

        xywh.append([
            (obj_x + obj_w / 2) * w,  # x of center
            (obj_y + obj_h / 2) * h,  # y of center
            w2,
            h2
        ])
        cls_conf.append(prob)
        cls_ids.append(cls_id)

    if len(xywh) == 0:
        xywh = np.array([]).reshape(0, 4)
        cls_conf = np.array([])
        cls_ids = np.array([])
    return np.array(xywh), np.array(cls_conf), np.array(cls_ids)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities, actions, action_datas, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i])
        action = actions[i]
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        label = label + " " + action
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1+t_size[1]+4),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            [255, 255, 255],
            2
        )

    return img

def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x)/u
