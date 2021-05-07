# Human tracker
# (c) 2020 ax Inc.

import sys
import time
import argparse

import cv2
import numpy as np
import math

import ailia

from sort.tracker import Tracker
from sort.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils import *

import keras
from keras.models import load_model

sys.path.append('../util')
from webcamera_utils import adjust_frame_size  # noqa: E402
from image_utils import load_image  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import check_file_existance  # noqa: E402

sys.path.append('../preprocess')
from normalize import normalize_keypoint,push_keypoint,TIME_RANGE
from pose_resnet_util import get_final_preds, get_affine_transform, compute, keep_aspect

# ======================
# Parameters 1
# ======================
SAVE_IMAGE_PATH = ""

ALGORITHM = ailia.POSE_ALGORITHM_LW_HUMAN_POSE

MODEL_LISTS = ['lw_human_pose', 'pose_resnet', 'none']
MODE_LISTS = ['default', 'dropout']

COCO_CATEGORY = [
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

FRAME_SKIP = False

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Fast and accurate human pose 2D-estimation.'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='The input video path. ' +
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='none', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-m', '--mode', metavar='MODE',
    default='default', choices=MODE_LISTS,
    help='mode lists: ' + ' | '.join(MODE_LISTS)
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
parser.add_argument(
    '-t', '--tiny',
    action='store_true',
    help='Use tiny model'
)
args = parser.parse_args()

# ======================
# Parameters 2
# ======================
MODEL_NAME = 'lightweight-human-pose-estimation'
if args.normal:
    WEIGHT_PATH = f'{MODEL_NAME}.onnx'
    MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'
else:
    WEIGHT_PATH = f'{MODEL_NAME}.opt.onnx'
    MODEL_PATH = f'{MODEL_NAME}.opt.onnx.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'

EX_WEIGHT_PATH = 'deep_sort.onnx'
EX_MODEL_PATH = 'deep_sort.onnx.prototxt'
EX_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deep_sort/'

POSE_MODEL_NAME = 'pose_resnet_50_256x192'
POSE_WEIGHT_PATH = f'{POSE_MODEL_NAME}.onnx'
POSE_MODEL_PATH = f'{POSE_MODEL_NAME}.onnx.prototxt'
POSE_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/pose_resnet/'

if args.tiny:
    DETECTOR_WEIGHT_PATH = 'yolov3-tiny.opt.onnx'
    DETECTOR_MODEL_PATH = 'yolov3-tiny.opt.onnx.prototxt'
    DETECTOR_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3-tiny/'
else:
    DETECTOR_WEIGHT_PATH = 'yolov3.opt.onnx'
    DETECTOR_MODEL_PATH = 'yolov3.opt.onnx.prototxt'
    DETECTOR_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3/'

# Deep sort model input
EX_INPUT_HEIGHT = 128
EX_INPUT_WIDTH = 64

# Metric parameters
MAX_COSINE_DISTANCE = 0.2  # threshold of matching object
NN_BUDGET = 100
MIN_CONFIDENCE = 0.3

# YOLO
THRESHOLD = 0.4
MIN_CONFIDENCE = 0.3
IOU = 0.45
POSE_THRESHOLD = 0.3

# Tracking Information
TRACKING_FRAMES = 120

# Entrance
ENTRANCE_X = 900
ENTRANCE_Y = 300
ENTRANCE_W = 200
ENTRANCE_H = 200

# Log
LOG_X = 1000
LOG_Y = 580
LOG_W = 280
LOG_H = 200

in_shop = {}
enter_log = []#"SHOP ENTER PM 9:00","SHOP ENTER PM 9:01"]

RESIZE_TO_720P = True

# ======================
# Utils
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def line(input_img, person, point1, point2):
    threshold = POSE_THRESHOLD
    if person.points[point1].score > threshold and\
       person.points[point2].score > threshold:
        color = hsv_to_rgb(255*point1/ailia.POSE_KEYPOINT_CNT, 255, 255)

        x1 = int(input_img.shape[1] * person.points[point1].x)
        y1 = int(input_img.shape[0] * person.points[point1].y)
        x2 = int(input_img.shape[1] * person.points[point2].x)
        y2 = int(input_img.shape[0] * person.points[point2].y)
        cv2.line(input_img, (x1, y1), (x2, y2), color, 5)


def display_result(input_img, person):
    line(input_img, person, ailia.POSE_KEYPOINT_NOSE,
            ailia.POSE_KEYPOINT_SHOULDER_CENTER)
    line(input_img, person, ailia.POSE_KEYPOINT_SHOULDER_LEFT,
            ailia.POSE_KEYPOINT_SHOULDER_CENTER)
    line(input_img, person, ailia.POSE_KEYPOINT_SHOULDER_RIGHT,
            ailia.POSE_KEYPOINT_SHOULDER_CENTER)

    line(input_img, person, ailia.POSE_KEYPOINT_EYE_LEFT,
            ailia.POSE_KEYPOINT_NOSE)
    line(input_img, person, ailia.POSE_KEYPOINT_EYE_RIGHT,
            ailia.POSE_KEYPOINT_NOSE)
    line(input_img, person, ailia.POSE_KEYPOINT_EAR_LEFT,
            ailia.POSE_KEYPOINT_EYE_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_EAR_RIGHT,
            ailia.POSE_KEYPOINT_EYE_RIGHT)

    line(input_img, person, ailia.POSE_KEYPOINT_ELBOW_LEFT,
            ailia.POSE_KEYPOINT_SHOULDER_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_ELBOW_RIGHT,
            ailia.POSE_KEYPOINT_SHOULDER_RIGHT)
    line(input_img, person, ailia.POSE_KEYPOINT_WRIST_LEFT,
            ailia.POSE_KEYPOINT_ELBOW_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_WRIST_RIGHT,
            ailia.POSE_KEYPOINT_ELBOW_RIGHT)

    line(input_img, person, ailia.POSE_KEYPOINT_BODY_CENTER,
            ailia.POSE_KEYPOINT_SHOULDER_CENTER)
    line(input_img, person, ailia.POSE_KEYPOINT_HIP_LEFT,
            ailia.POSE_KEYPOINT_BODY_CENTER)
    line(input_img, person, ailia.POSE_KEYPOINT_HIP_RIGHT,
            ailia.POSE_KEYPOINT_BODY_CENTER)

    line(input_img, person, ailia.POSE_KEYPOINT_KNEE_LEFT,
            ailia.POSE_KEYPOINT_HIP_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_ANKLE_LEFT,
            ailia.POSE_KEYPOINT_KNEE_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_KNEE_RIGHT,
            ailia.POSE_KEYPOINT_HIP_RIGHT)
    line(input_img, person, ailia.POSE_KEYPOINT_ANKLE_RIGHT,
            ailia.POSE_KEYPOINT_KNEE_RIGHT)


def pose_recognition(box,input_image,pose,detector):
    if args.arch=="lw_human_pose":
        bbox_xywh, cls_conf, cls_ids = get_detector_result_lw_human_pose(pose, input_image.shape[0], input_image.shape[1], get_all=True)

        idx = -1
        min_d = 32768

        for i in range(pose.get_object_count()):
            target=xywh_to_xyxy(bbox_xywh[i], input_image.shape[0], input_image.shape[1])
            d=math.sqrt((target[0]-box[0])**2 + (target[1]-box[1])**2)
            if d < min_d:
                min_d = d
                idx = i

        if idx == -1:
            return None

        person = pose.get_object_pose(idx)

    if args.arch=="pose_resnet":
        bbox_xywh, cls_conf, cls_ids = get_detector_result(detector, input_image.shape[0], input_image.shape[1])
        px1,py1,px2,py2 = keep_aspect((box[0],box[1]),(box[2],box[3]),input_image,pose)
        crop_img = input_image[py1:py2,px1:px2,:]
        offset_x = px1/input_image.shape[1]
        offset_y = py1/input_image.shape[0]
        scale_x = crop_img.shape[1]/input_image.shape[1]
        scale_y = crop_img.shape[0]/input_image.shape[0]
        detections = compute(pose,crop_img,offset_x,offset_y,scale_x,scale_y)
        #cv2.imwrite("crop.png",crop_img)
        person=detections
    
    if args.arch=="none":
        person=None

    return person


def action_recognition(input_image,position,idx_frame):
    action="-"

    w = input_image.shape[1]
    h = input_image.shape[0]

    cnt = 0
    d = 0
    for i in range(len(position)-1):
        frame1 = int(position[i][0])
        x1 = int(position[i][1])
        y1 = int(position[i][2])
        w1 = int(position[i][3])
        h1 = int(position[i][4])
        p1 = (position[i][5])

        j = len(position)-1 #last point
        frame2 = int(position[j][0])
        x2 = int(position[j][1])
        y2 = int(position[j][2])
        w2 = int(position[j][3])
        h2 = int(position[j][4])
        p2 = (position[j][5])

        if frame1<idx_frame-TRACKING_FRAMES:
            continue
        
        if p1:
            #for i in range(ailia.POSE_KEYPOINT_CNT):
            #    if p1.points[i].score>=0.1 and p2.points[i].score>=0.1:
            #        x=p1.points[i].x-p2.points[i].x
            #        y=p1.points[i].y-p2.points[i].y
            #        d = d + math.sqrt(x**2 + y**2)
            #        cnt = cnt + 1
            
            x1 = p1.points[ailia.POSE_KEYPOINT_BODY_CENTER].x
            y1 = p1.points[ailia.POSE_KEYPOINT_BODY_CENTER].y

            x2 = p2.points[ailia.POSE_KEYPOINT_BODY_CENTER].x
            y2 = p2.points[ailia.POSE_KEYPOINT_BODY_CENTER].y

            d = d + math.sqrt((x1-x2)**2 + (y1-y2)**2)
            cnt = cnt + 1
        else:
            #x1, y1 = projection(x1,y1,w1,h1,w,h)
            #x2, y2 = projection(x2,y2,w2,h2,w,h)

            x1=x1/w
            y1=y1/h
            x2=x2/w
            y2=y2/h

            #from body center
            #x1=fx1
            #y1=fy1
            #x2=fx2
            #y2=fy2

            d = d + math.sqrt((x1-x2)**2 + (y1-y2)**2)
            cnt = cnt + 1
    
    if cnt>=TRACKING_FRAMES/2:
        d=d/cnt
        score=str(int(d*10000)/10000)
        if d>=0.01:
            action="WALK "+score
        else:
            action="STAND "+score
    else:
        action="-"
    action=""
    
    return action


def resize(img, size=(EX_INPUT_WIDTH, EX_INPUT_HEIGHT)):
    return cv2.resize(img.astype(np.float32), size)


def projection(x,y,w,h,iw,ih):
    y = y + h/2
    return x,y


def display_position(input_img,position,idx_frame):
    w = input_img.shape[1]
    h = input_img.shape[0]

    d = 4

    cv2.rectangle(input_img, (0, 0), (int(w/d), int(h/d)), (0, 0, 0), thickness=-1)

    for id in position:
        color = compute_color_for_labels(id)
        for i in range(len(position[id])-1):
            frame1 = int(position[id][i][0])
            x1 = int(position[id][i][1])
            y1 = int(position[id][i][2])
            w1 = int(position[id][i][3])
            h1 = int(position[id][i][4])

            frame2 = int(position[id][i][0])
            x2 = int(position[id][i+1][1])
            y2 = int(position[id][i+1][2])
            w2 = int(position[id][i+1][3])
            h2 = int(position[id][i+1][4])

            if frame1<idx_frame-TRACKING_FRAMES:
                continue
                
            x1, y1 = projection(x1,y1,w1,h1,w,h)
            x2, y2 = projection(x2,y2,w2,h2,w,h)

            x1 = int(x1/d)
            y1 = int(y1/d)
            x2 = int(x2/d)
            y2 = int(y2/d)

            cv2.line(input_img, (x1, y1), (x2, y2), color, 5)

def display_entrance(img):
    color = (0,0,255)
    top_left = (ENTRANCE_X, ENTRANCE_Y)
    bottom_right = (ENTRANCE_X + ENTRANCE_W, ENTRANCE_Y + ENTRANCE_H)
    cv2.rectangle(img, top_left, bottom_right, color, 4)

def display_log(img,enter_log):
    x = LOG_X + 8
    y = LOG_Y + 32

    cv2.rectangle(img, (LOG_X, LOG_Y), (LOG_X + LOG_W, LOG_Y + LOG_H), (0,0,0), thickness=-1)

    for log in enter_log:
        text_position = (x+4, y-8)

        # update image
        color = (255,255,255)#hsv_to_rgb(256 * obj.category / len(category), 255, 255)
        fontScale = 0.5

        cv2.putText(
            img,
            log,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )

        y=y+16

def entrance_check(px, py, id, time_stamp):
    if px>=ENTRANCE_X and px<=ENTRANCE_X+ENTRANCE_W and py>=ENTRANCE_Y and py<=ENTRANCE_Y+ENTRANCE_H:
        if not(id in in_shop):
            in_shop[id]={"x":px,"y":py,"enter":False,"leave":False}
        else:
            if not in_shop[id]["enter"]:
                if in_shop[id]["x"]>=ENTRANCE_X+ENTRANCE_W/2 and px<ENTRANCE_X+ENTRANCE_W/2:
                    in_shop[id]["enter"]=True
                    enter_log.append("SHOP ENTER "+time_stamp+" ID "+str(id))
            if not in_shop[id]["leave"]:
                if in_shop[id]["x"]<ENTRANCE_X+ENTRANCE_W/2 and px>=ENTRANCE_X+ENTRANCE_W/2:
                    in_shop[id]["leave"]=True
                    enter_log.append("SHOP LEAVE "+time_stamp+" ID "+str(id))
            in_shop[id]["x"]=px
            in_shop[id]["y"]=py

# ======================
# Main functions
# ======================

def recognize_from_video():
    if args.video == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    else:
        if check_file_existance(args.video):
            capture = cv2.VideoCapture(args.video)
    
    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    if FRAME_SKIP:
        action_recognize_fps = 10
    else:
        action_recognize_fps = frame_rate

    if args.savepath is not None:
        if RESIZE_TO_720P:
            size = (1280, 720)
        else:
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(args.savepath, fmt, action_recognize_fps, size)

    # pose estimation
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    if args.arch=="lw_human_pose":
        pose = ailia.PoseEstimator(
            MODEL_PATH, WEIGHT_PATH, env_id=env_id, algorithm=ALGORITHM
        )

        detector = None
    if args.arch=="pose_resnet" or args.arch=="none":
        detector = ailia.Detector(
            DETECTOR_MODEL_PATH,
            DETECTOR_WEIGHT_PATH,
            len(COCO_CATEGORY),
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
            env_id=env_id
        )

        pose = ailia.Net(POSE_MODEL_PATH, POSE_WEIGHT_PATH, env_id=env_id)

    # tracker class instance
    extractor = ailia.Net(EX_MODEL_PATH, EX_WEIGHT_PATH, env_id=env_id)
    metric = NearestNeighborDistanceMetric(
        "cosine", MAX_COSINE_DISTANCE, NN_BUDGET
    )
    tracker = Tracker(
        metric,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3
    )

    action_data = {}
    position = {}

    frame_nb = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    idx_frame = 0

    while(True):
        s = format(int(idx_frame/120), '02d')
        time_stamp = "PM 9:"+s
        
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if (not ret) or (frame_nb>=1 and idx_frame>=frame_nb):
            break

        if FRAME_SKIP:
            mod = int(frame_rate/action_recognize_fps)
            if mod>=1:
                if idx_frame%mod != 0:
                    idx_frame = idx_frame + 1
                    continue
        
        if RESIZE_TO_720P:
            frame = cv2.resize(frame, (1280, 720))

        input_image, input_data = adjust_frame_size(
            frame, frame.shape[0], frame.shape[1],
        )
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2BGRA)

        # inferece
        if args.arch=="lw_human_pose":
            _ = pose.compute(input_data)
        else:
            detector.compute(input_data, THRESHOLD, IOU)

        # deepsort format
        h, w = input_image.shape[0], input_image.shape[1]
        if args.arch=="lw_human_pose":
            bbox_xywh, cls_conf, cls_ids = get_detector_result_lw_human_pose(pose, h, w)
        if args.arch=="pose_resnet" or args.arch=="none":
            bbox_xywh, cls_conf, cls_ids = get_detector_result(detector, h, w)
        
        mask = cls_ids == 0
        bbox_xywh = bbox_xywh[mask]

        # bbox dilation just in case bbox too small,
        # delete this line if using a better pedestrian detector
        if args.arch=="pose_resnet" or args.arch=="none":
            bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

        # do tracking
        img_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
            img_crops.append(input_image[y1:y2, x1:x2])

        if img_crops:
            # preprocess
            img_batch = np.concatenate([
                normalize_image(resize(img), 'ImageNet')[np.newaxis, :, :, :]
                for img in img_crops
            ], axis=0).transpose(0, 3, 1, 2)

            # TODO better to pass a batch at once
            # features = extractor.predict(img_batch)
            features = []
            for img in img_batch:
                features.append(extractor.predict(img[np.newaxis, :, :, :])[0])
            features = np.array(features)
        else:
            features = np.array([])

        bbox_tlwh = xywh_to_tlwh(bbox_xywh)
        detections = [
            Detection(bbox_tlwh[i], conf, features[i])
            for i, conf in enumerate(cls_conf) if conf > MIN_CONFIDENCE
        ]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        nms_max_overlap = 1.0
        indices = non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        tracker.predict()
        tracker.update(detections)

        # update bbox identities
        outputs = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = tlwh_to_xyxy(box, h, w)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        # action detection
        actions = []
        persons = []
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            for i, box in enumerate(bbox_xyxy):
                id = identities[i]

                if not(id in position):
                    position[id] = []

                 # action recognition
                person = pose_recognition(box, input_image, pose, detector)
                persons.append(person)

                pos = [idx_frame,(box[0]+box[2])/2,(box[1]+box[3])/2,(box[2]-box[0]),(box[3]-box[1]),person]
                position[id].append(pos)

                action = action_recognition(input_image, position[id], idx_frame)
                actions.append(action)

                # log
                px = (box[0] + box[2])/2
                py = box[3]
                entrance_check(px, py, id, time_stamp)
                
        # draw box for visualization
        if len(outputs) > 0:
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            frame = draw_boxes(input_image, bbox_xyxy, identities, actions, None)

            for bb_xyxy in bbox_xyxy:
                bbox_tlwh.append(xyxy_to_tlwh(bb_xyxy))

        # draw skelton
        for person in persons:
            if person!=None:
                display_result(input_image, person)

        # draw flow
        display_position(input_image,position,idx_frame)

        # draw entrance
        display_entrance(input_image)

        # draw log
        display_log(input_image, enter_log)

        #for i in range(len(flows)):
        #    input_image[0:19,TIME_RANGE*i:TIME_RANGE*(i+1),0:3]=flows[i][0:19,0:TIME_RANGE,0:3]

        cv2.imshow('frame', input_image)

        if writer is not None:
            writer.write(input_image)

            # show progress
            if idx_frame == "0":
                print()
            print("\r" + str(idx_frame + 1) + " / " + str(frame_nb) ,end="")
            if idx_frame == frame_nb - 1:
                print()
            
        idx_frame = idx_frame + 1

    if writer is not None:
        writer.release()

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    if args.arch=="lw_human_pose":
        check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
        check_and_download_models(EX_WEIGHT_PATH, EX_MODEL_PATH, EX_REMOTE_PATH)
    else:
        check_and_download_models(POSE_WEIGHT_PATH, POSE_MODEL_PATH, POSE_REMOTE_PATH)
        check_and_download_models(DETECTOR_WEIGHT_PATH, DETECTOR_MODEL_PATH, DETECTOR_REMOTE_PATH)
        check_and_download_models(EX_WEIGHT_PATH, EX_MODEL_PATH, EX_REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        print("video not found")


if __name__ == '__main__':
    main()
