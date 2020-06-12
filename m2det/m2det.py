import torch
import numpy as np
import cv2
import time

from configs.CC import Config
# from layers.functions import Detect, PriorBox
import torch.backends.cudnn as cudnn
import sys
sys.path.insert(0, '/media/prolley/M2/Projects_2/Ailia-SDK/ailia_1_22_0/tools/onnx/')
import onnx_optimizer
import onnx2prototxt
import math
import os.path
import argparse
import m2det
import pathlib

# ======================
# Parameters
# ======================

WEIGHT_PATH = './m2det.onnx'
MODEL_PATH = './m2det.onnx.prototxt'
# REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3/'

IMAGE_PATH = 'couple.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 448  # for video mode
IMAGE_WIDTH = 448  # for video mode

COCO_CATEGORY = ['__background__'] + [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
THRESHOLD = 0.4
IOU = 0.45
KEEP_PER_CLASS = 10

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Yolov3 model'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=IMAGE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='The input video path. ' +
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
args = parser.parse_args()

# ======================
# Secondaty Functions
# ======================

def load_image(image_path):
    if pathlib.Path(image_path).exists():
        image = cv2.imread(image_path)
    else:
        print(f'[ERROR] {image_path} not found.')
        sys.exit()
    return image

# from /utils/nms
def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

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

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def _preprocess(img, resize=512, rgb_means=(104,117,123), swap=(2, 0, 1)):
    interp_method = cv2.INTER_LINEAR
    img = cv2.resize(np.array(img), 
                     (resize, resize),
                     interpolation = interp_method).astype(np.float32)
    img -= rgb_means
    img = img.transpose(swap)
    return torch.from_numpy(img)

def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127

base = int(np.ceil(pow(len(COCO_CATEGORY), 1. / 3)))
COLORS = [_to_color(x, base) for x in range(len(COCO_CATEGORY))]

# from m2det.py
def draw_detection(im, bboxes, scores, cls_inds, fps, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = int(cls_inds[i])
        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      COLORS[cls_indx], thick)
        mess = '%s: %.3f' % (COCO_CATEGORY[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                    0, 1e-3 * h, COLORS[cls_indx], thick // 3)
        if fps >= 0:
            cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15), 0, 2e-3 * h, (255, 255, 255), thick // 2)

    return imgcv

# ======================
# Main functions
# ======================

def recognize_objects(filename):
    import onnxruntime as ort
    
    # load model
    ort_session = ort.InferenceSession(WEIGHT_PATH)
    
    # initial preprocesses
    
    labels = tuple(['__background__'] + COCO_CATEGORY)
    image = load_image(filename)
    w, h, c = image.shape
    img = _preprocess(image).unsqueeze(0)

    # move img to gpu
    cuda=True
    if cuda:
        img = img.cuda()
        
    # feedforward
    out = ort_session.run(None, {'input.1': img.cpu().numpy()})
    boxes, scores =  (torch.Tensor(out[0]), torch.Tensor(out[1]))
    
   
    scale = torch.Tensor([w,h,w,h])
    boxes = (boxes[0]*scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    allboxes = []
    
    # filter boxes for every class
    for j in range(1, len(COCO_CATEGORY)):
        inds = np.where(scores[:,j] > THRESHOLD)[0]
        if len(inds) == 0:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
        # rank ordered iou
        keep = nms(c_dets, IOU) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
        keep = keep[:KEEP_PER_CLASS]
        c_dets = c_dets[keep, :]
        allboxes.extend([_.tolist()+[j] for _ in c_dets])
    allboxes = np.array(allboxes)
    
#     split boxes and scores
    boxes = allboxes[:,:4]
    scores = allboxes[:,4]
    cls_inds = allboxes[:,5]
    
    print('\n'.join(['pos:{}, ids:{}, score:{:.3f}'.format('(%.1f,%.1f,%.1f,%.1f)' % (o[0],o[1],o[2],o[3]) \
            ,labels[int(oo)],ooo) for o,oo,ooo in zip(boxes,cls_inds,scores)]))
    
    # fps = 1.0 / float(loop_time) if cam >= 0 or video else -1
    fps = -1
    
    # show image
    im2show = draw_detection(image, boxes, scores, cls_inds, fps)
    cv2.imshow('demo', im2show)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

if __name__=='__main__':
    recognize_objects(args.input)