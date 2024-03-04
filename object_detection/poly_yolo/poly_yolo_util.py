import os

import math
import cv2 as cv
import numpy as np

from scipy.special import expit

import ailia

np.set_printoptions(precision=3, suppress=True)
ANGLE_STEP  = 15 #that means Poly-YOLO will detect 360/15=24 vertices per polygon at max
NUM_ANGLES3  = int(360 // ANGLE_STEP * 3)
NUM_ANGLES  = int(360 // ANGLE_STEP)

grid_size_multiplier = 4 #that is resolution of the output scale compared with input. So it is 1/4
anchor_mask = [[0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8]] #that should be optimized
anchors_per_level = 9 #single scale and nine anchors

def gather(a, nms_indexs,shape=None):
    if len(nms_indexs) ==0:
        return np.reshape(nms_indexs,shape)
    else:
        return np.take_along_axis(a,nms_indexs,0)

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw = image.shape[1]
    ih = image.shape[0]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    cvi = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cvi = cv.resize(cvi, (nw, nh), interpolation=cv.INTER_CUBIC)
    dx = int((w - nw) // 2)
    dy = int((h - nh) // 2)
    new_image = np.zeros((h, w, 3), dtype='uint8')
    new_image[...] = 128
    if nw <= w and nh <= h:
        new_image[dy:dy + nh, dx:dx + nw, :] = cvi
    else:
        new_image = cvi[-dy:-dy + h, -dx:-dx + w, :]

    return new_image.astype('float32') / 255.0

class YOLO(object):
    _defaults = {
        "model":None,
        "model_path": 'poly_yolo.onnx',
        "anchors_path": 'yolo_anchors.txt',
        "classes_path": 'yolo_classes.txt',
        "score": 0.2,
        "iou": 0.4,
        "model_image_size": (416,832),
        "gpu_num": 1,
    }


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


    def detect_image(self, image):

        def correct_polygons(polygons_x, polygons_y, polygons_confidence):
            polygons = np.concatenate([polygons_x,polygons_y,polygons_confidence],-1)
            return polygons
        
        def correct_boxes(box_xy, box_wh, input_shape, image_shape):
            box_yx = box_xy[..., ::-1]
            box_hw = box_wh[..., ::-1]
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape
            box_yx = (box_yx - offset) * scale
            box_hw *= scale

            box_mins = box_yx - (box_hw / 2.)
            box_maxes = box_yx + (box_hw / 2.)

            boxes = np.concatenate([
                box_mins[..., 0:1],  # y_min
                box_mins[..., 1:2],  # x_min
                box_maxes[..., 0:1],  # y_max
                box_maxes[..., 1:2]  # x_max
            ],-1)
            return boxes

 
        def head(feats, anchors, num_classes, input_shape, calc_loss=False):
            
            num_anchors = anchors_per_level
            # Reshape to batch, height, width, num_anchors, box_params.
            anchors_tensor = np.reshape(anchors, [1, 1, 1, num_anchors, 2])
            
            grid_shape = feats.shape[1:3]  # height, width
            grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                            [1, grid_shape[1], 1, 1])
            grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                            [grid_shape[0], 1, 1, 1])
            grid = np.concatenate([grid_x, grid_y], axis=-1)
            
            feats = feats.reshape([-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5 + NUM_ANGLES3])
            
            box_xy = (expit(feats[..., :2]) + grid)/ np.array(grid_shape)[...,::-1]
            box_wh = np.exp(feats[..., 2:4]) * anchors_tensor / input_shape[...,::-1]
            
            box_confidence      = expit(feats[..., 4:5])
            box_class_probs     = expit(feats[..., 5:5 + num_classes])
            polygons_confidence = expit(feats[..., 5+num_classes+2:5+num_classes+NUM_ANGLES3:3])
            polygons_x = np.exp(feats[..., 5 + num_classes:num_classes + 5 + NUM_ANGLES3:3])
            
            dx = np.square(anchors_tensor[..., 0:1] / 2)
            dy = np.square(anchors_tensor[..., 1:2] / 2)
            d = np.sqrt(dx + dy)
            a = np.power(input_shape[::-1], 2)
            b= np.sum(a)
            diagonal = np.sqrt(b)
            polygons_x = polygons_x * d / diagonal
            
            polygons_y = feats[..., 5 + num_classes + 1:num_classes + 5 + NUM_ANGLES3:3]
            polygons_y = expit(polygons_y)
            return box_xy, box_wh, box_confidence, box_class_probs, polygons_x, polygons_y, polygons_confidence

        def boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
            box_xy, box_wh, box_confidence, box_class_probs, polygons_x, polygons_y, polygons_confidence = head(feats, anchors, num_classes, input_shape)
            boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)[0]

            boxes *= np.concatenate([image_shape,image_shape],-1)
            boxes = np.array([boxes.reshape((-1,4))])

            box_scores = [[box_confidence, box_class_probs]]

            box_scores = np.array(box_scores[0][0]) * np.array(box_scores[0][1])
            box_scores = [box_scores.reshape([-1, num_classes])]

            polygons = correct_polygons(polygons_x,polygons_y,polygons_confidence)

            polygons = np.array([polygons.reshape((-1,NUM_ANGLES3))] )
 
            return boxes, box_scores, polygons

        if self.model_image_size != (None, None):
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            print('THE functionality is not implemented!')

        image_data = np.expand_dims(boxed_image, 0)  # Add batch dimension.

        model_inference = self.model.run(image_data)
        model_inference = np.array(model_inference)[0][0]
        
        image_shape = np.array([image.shape[0],image.shape[1]])
        input_shape = np.array(model_inference.shape)[0:2] * grid_size_multiplier

        num_classes = len(self.class_names)
        score_threshold=self.score
        iou_threshold=self.iou
        
        feats = np.array([model_inference])
        anchors = self.anchors[anchor_mask[0]]
       
        boxes, box_scores, polygons = boxes_and_scores(feats,
                                                       anchors[anchor_mask[0]],
                                                       num_classes, input_shape,
                                                       image_shape)
 
        #infacens args
        boxes_ = []
        scores_ = []
        classes_ = []
        polygons_ = []


        boxes    = np.concatenate(boxes   ,0)
        box_scores    = np.concatenate(box_scores   ,0)

        mask1 = box_scores >= score_threshold
        box_scores >= score_threshold

        polygons    = np.concatenate(polygons   ,0)
        for c in range(num_classes):
            # TODO: use keras backend instead of tf.

            class_boxes = boxes[mask1[:,c]]

            score = box_scores[:,c]
            score = score[mask1[:,c]]
 
            iou_threshold = 0.5
            nms_index = np.array(nms(class_boxes,score,iou_threshold))
            nms_indexs = []
            for n in nms_index:
                nms_indexs.append([n])
            nms_indexs = np.array(nms_indexs,dtype=np.int32)
            
            if len(class_boxes) > 0:
                class_boxes = np.array(gather(np.array(class_boxes), nms_indexs))
            else:
                class_boxes = np.reshape(np.array([]),(0,4))
            boxes_.append(class_boxes)
            
            if len(score) > 0:
                class_box_scores = gather(score, nms_index)
            else:
                class_box_scores = []
            scores_.append(np.array(class_box_scores))

            if len(class_box_scores) >0:
                classes = np.ones_like(np.array(class_box_scores,dtype=np.int32)) * c
            else:
                classes = []
            classes_.append(classes)

            polygon = polygons[mask1[:,c]]
            class_polygons = gather(polygon,nms_indexs,(0,72))
            polygons_.append(class_polygons)


        out_boxes   = np.concatenate(boxes_  ,0)
        out_scores  = np.concatenate(scores_ ,0)

        out_classes = np.concatenate(classes_,0)
        polygons    = np.concatenate(polygons_   ,0)
 
        

        for b in range(0, out_boxes.shape[0]):
            cy = (out_boxes[b, 0] + out_boxes[b, 2]) // 2
            cx = (out_boxes[b, 1] + out_boxes[b, 3]) // 2
            diagonal = np.sqrt(np.power(out_boxes[b, 3] - out_boxes[b, 1], 2.0) + np.power(out_boxes[b, 2] - out_boxes[b, 0], 2.0))
            for i in range(0, NUM_ANGLES):
                x1 = cx - math.cos(math.radians((polygons[b, i+NUM_ANGLES] + i) / NUM_ANGLES * 360)) * polygons[b, i] *diagonal# scale[1]
                y1 = cy - math.sin(math.radians((polygons[b, i+NUM_ANGLES] + i) / NUM_ANGLES * 360)) * polygons[b, i] *diagonal# scale[0]
                polygons[b, i]            = x1
                polygons[b, i+NUM_ANGLES] = y1

        return out_boxes, out_scores, out_classes, polygons

    def close_session(self):
        self.sess.close()


