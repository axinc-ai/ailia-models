import sys
import time
import argparse
import os
import re

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from webcamera_utils import adjust_frame_size  # noqa: E402
from image_utils import load_image, draw_result_on_img  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import check_file_existance  # noqa: E402
from detector_utils import hsv_to_rgb # noqa: E402C

import matplotlib.pyplot as plt

# ======================
# PARAMETERS
# ======================

MODEL_LISTS = ['arcface', 'arcface_mixed_90_82', 'arcface_mixed_90_99', 'arcface_mixed_eq_90_89']

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/arcface/"

IMG_PATH_1 = 'correct_pair_1.jpg'
IMG_PATH_2 = 'correct_pair_2.jpg'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# the threshold was calculated by the `test_performance` function in `test.py`
# of the original repository
THRESHOLD = 0.25572845
# THRESHOLD = 0.45 # for mixed model

# face detection
FACE_MODEL_LISTS = ['yolov3', 'blazeface', 'yolov3-mask', 'mobilenet-mask']

YOLOV3_FACE_THRESHOLD = 0.4
YOLOV3_FACE_IOU = 0.45

BLAZEFACE_INPUT_IMAGE_HEIGHT = 128
BLAZEFACE_INPUT_IMAGE_WIDTH = 128

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Determine if the person is the same from two facial images.'
)
parser.add_argument(
    '-i', '--inputs', metavar='IMAGE',
    nargs=2,
    default=[IMG_PATH_1, IMG_PATH_2],
    help='Two image paths for calculating the face match.'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='The input video path. ' +
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='arcface', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-f', '--face', metavar='FACE_ARCH',
    default='yolov3', choices=FACE_MODEL_LISTS,
    help='dace detection model lists: ' + ' | '.join(FACE_MODEL_LISTS)
)
parser.add_argument(
    '-t', '--threshold', type=float, default=THRESHOLD,
    help='Similality threshold for identification'
) 
args = parser.parse_args()

WEIGHT_PATH = args.arch+'.onnx'
MODEL_PATH = args.arch+'.onnx.prototxt'

if args.face=="yolov3":
    FACE_WEIGHT_PATH = 'yolov3-face.opt.onnx'
    FACE_MODEL_PATH = 'yolov3-face.opt.onnx.prototxt'
    FACE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3-face/'
elif args.face=="yolov3-mask":
    FACE_WEIGHT_PATH = 'face-mask-detection-yolov3-tiny.opt.obf.onnx'
    FACE_MODEL_PATH = 'face-mask-detection-yolov3-tiny.opt.onnx.prototxt'
    FACE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/face-mask-detection/'
elif args.face=="mobilenet-mask":
    FACE_WEIGHT_PATH = 'face-mask-detection-mb2-ssd-lite.obf.onnx'
    FACE_MODEL_PATH = 'face-mask-detection-mb2-ssd-lite.onnx.prototxt'
    FACE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/face-mask-detection/'
else:
    FACE_WEIGHT_PATH = 'blazeface.onnx'
    FACE_MODEL_PATH = 'blazeface.onnx.prototxt'
    FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
    sys.path.append('../../object-detection/blazeface')
    from blazeface_utils import *

# ======================
# Utils
# ======================
def preprocess_image(image, input_is_bgr=False):
    # (ref: https://github.com/ronghuaiyang/arcface-pytorch/issues/14)
    # use origin image and fliped image to infer,
    # and concat the feature as the final feature of the origin image.
    if input_is_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if "eq_" in args.arch:
        image = cv2.equalizeHist(image)
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    return image / 127.5 - 1.0  # normalize


def prepare_input_data(image_path):
    image = load_image(
        image_path,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
        rgb=False,
        normalize_type='None'
    )
    return preprocess_image(image)


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


class FaceTrack():
    def __init__(self, id, fe, image, frame_no):
        self.id = id
        self.fe = [fe]
        self.image = [image]
        self.frame_no = [frame_no]
        self.TRACK_T = 16
        self.REMOVE_T = 80
        self.score = 0
    
    def update(self,fe,image,score,frame_no):
        self.fe.append(fe)
        self.image.append(image)
        self.frame_no.append(frame_no)
        self.score=score
    
    def pop(self,frame_no):
        if len(self.frame_no)>=self.TRACK_T:
            self.fe.pop(0)
            self.image.pop(0)
            self.frame_no.pop(0)

        if len(self.frame_no)>=1:
            if frame_no - self.frame_no[0] >= self.REMOVE_T:
                self.fe.pop(0)
                self.image.pop(0)
                self.frame_no.pop(0)


def face_identification(tracks,net,detections,frame_no):
    BATCH_SIZE = net.get_input_shape()[0]

    for i in range(len(detections)):
        resized_frame = detections[i]["resized_frame"]

        # prepare target face and input face
        input_frame = preprocess_image(resized_frame, input_is_bgr=True)
        if BATCH_SIZE == 4:
            input_data = np.concatenate([input_frame, input_frame], axis=0)
        else:
            input_data = input_frame

        # inference
        preds_ailia = net.predict(input_data)

        # postprocessing
        if BATCH_SIZE == 4:
            fe_1 = np.concatenate([preds_ailia[0], preds_ailia[1]], axis=0)
            fe_2 = np.concatenate([preds_ailia[2], preds_ailia[3]], axis=0)
        else:
            fe_1 = np.concatenate([preds_ailia[0], preds_ailia[1]], axis=0)
            fe_2 = fe_1
        
        detections[i]["fe"] = fe_2
    
    score_matrix = np.zeros((len(detections),len(tracks)))
    for i in range(len(detections)):
        for j in range(len(tracks)):
            max_sim = 0
            avg_sim = 0
            for k in range(len(tracks[j].fe)):
                sim = cosin_metric(detections[i]["fe"], tracks[j].fe[k])
                avg_sim = avg_sim + sim
                if max_sim < sim:
                    max_sim = sim
            score_matrix[i,j] = max_sim
            if len(tracks[j].fe)>=1:
                avg_sim = avg_sim / len(tracks[j].fe)
            score_matrix[i,j] = avg_sim

    for i in range(len(detections)):
        score_sim = 0
        id_sim = 0
        det_sim = 0
        for j in range(len(detections)):
            for k in range(len(tracks)):
                if score_sim < score_matrix[j,k]:
                    score_sim = score_matrix[j,k]
                    det_sim = j
                    id_sim = k
        detections[det_sim]["id_sim"] = id_sim
        detections[det_sim]["score_sim"] = score_sim
        for j in range(len(detections)):
            for k in range(len(tracks)):
                if j==det_sim or k==id_sim:
                    score_matrix[j,k]=0

    for i in range(len(tracks)):
        tracks[i].score=0

    for i in range(len(detections)):
        # identification
        if detections[i]["score_sim"] < args.threshold:
            id_sim = len(tracks)
            fe_obj=FaceTrack(id_sim,detections[i]["fe"],detections[i]["resized_frame"],frame_no)
            tracks.append(fe_obj)
            detections[i]["score_sim"] = 0
            detections[i]["id_sim"] = id_sim
        else:
            tracks[detections[i]["id_sim"]].update(detections[i]["fe"],detections[i]["resized_frame"],detections[i]["score_sim"],frame_no)

    for i in range(len(tracks)):
        tracks[i].pop(frame_no)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

# ======================
# Main functions
# ======================
def compare_images():
    # prepare input data
    imgs_1 = prepare_input_data(args.inputs[0])
    imgs_2 = prepare_input_data(args.inputs[1])
    imgs = np.concatenate([imgs_1, imgs_2], axis=0)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    BATCH_SIZE = net.get_input_shape()[0]

    # inference
    print('Start inference...')
    if BATCH_SIZE==2:
        shape = net.get_output_shape()
        shape = (4,shape[1])
        preds_ailia = np.zeros(shape)
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            if BATCH_SIZE==4:
                preds_ailia = net.predict(imgs)
            else:
                preds_ailia[0:2] = net.predict(imgs[0:2])
                preds_ailia[2:4] = net.predict(imgs[2:4])
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        if BATCH_SIZE==4:
            preds_ailia = net.predict(imgs)
        else:
            preds_ailia[0:2] = net.predict(imgs[0:2])
            preds_ailia[2:4] = net.predict(imgs[2:4])

    # postprocessing
    fe_1 = np.concatenate([preds_ailia[0], preds_ailia[1]], axis=0)
    fe_2 = np.concatenate([preds_ailia[2], preds_ailia[3]], axis=0)
    sim = cosin_metric(fe_1, fe_2)

    print(f'Similarity of ({args.inputs[0]}, {args.inputs[1]}) : {sim:.3f}')
    if args.threshold > sim:
        print('They are not the same face!')
    else:
        print('They are the same face!')


def compare_video():
    # prepare base image
    tracks = []

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # detector initialize
    if args.face=="yolov3":
        detector = ailia.Detector(
            FACE_MODEL_PATH,
            FACE_WEIGHT_PATH,
            1,
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
            env_id=env_id
        )
    elif args.face=="yolov3-mask":
        detector = ailia.Detector(
            FACE_MODEL_PATH,
            FACE_WEIGHT_PATH,
            2,
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
            env_id=env_id
        )
    elif args.face=="mobilenet-mask":
        detector = ailia.Detector(
            FACE_MODEL_PATH,
            FACE_WEIGHT_PATH,
            2,
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_SSD,
            env_id=env_id
        )
    else:
        detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=env_id)

    # web camera
    if args.video == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[Error] webcamera not found")
            sys.exit(1)
    else:
        if check_file_existance(args.video):
            capture = cv2.VideoCapture(args.video)

    # ui buffer
    ui = np.zeros((1,1,3), np.uint8)
    frame_no = 0

    # inference loop
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        h, w = frame.shape[0], frame.shape[1]
        if ui.shape[0]==1:
            ui = np.zeros((h,int(w * 1.5),3), np.uint8)

        # detect face
        org_detections = []
        if args.face=="yolov3" or args.face=="yolov3-mask" or args.face=="mobilenet-mask":
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            detector.compute(img, YOLOV3_FACE_THRESHOLD, YOLOV3_FACE_IOU)
            count = detector.get_object_count()
            for idx in range(count):
                obj = detector.get_object(idx)
                org_detections.append(obj)
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(img, (BLAZEFACE_INPUT_IMAGE_WIDTH, BLAZEFACE_INPUT_IMAGE_HEIGHT))
            image = image.transpose((2, 0, 1))  # channel first
            image = image[np.newaxis, :, :, :]  # (batch_size, channel, h, w)
            input_data = image / 127.5 - 1.0

            # inference
            preds_ailia = detector.predict([input_data])

            # postprocessing
            org_detections = []
            blaze_face_detections = postprocess(preds_ailia)
            for idx in range(blaze_face_detections):
                obj = blaze_face_detections[idx]
                if len(obj)==0:
                    continue
                d = obj[0]
                obj = ailia.DetectorObject(
                    category = 0,
                    prob = 1.0,
                    x = d[1],
                    y = d[0],
                    w = d[3]-d[1],
                    h = d[2]-d[0] )
                
                org_detections.append(obj)

            count = len(org_detections)

        # remove overwrapped detection
        det = []
        keep = []
        iou_threshold = 0.25
        for idx in range(len(org_detections)):
            obj = org_detections[idx]
            is_keep=True
            for idx2 in range(len(det)):
                if not keep[idx2]:
                    continue
                box_a = [w*det[idx2].x,h*det[idx2].y,w*(det[idx2].x+det[idx2].w),h*(det[idx2].y+det[idx2].h)]
                box_b = [w*obj.x,h*obj.y,w*(obj.x+obj.w),h*(obj.y+obj.h)]
                iou = bb_intersection_over_union(box_a,box_b)
                if iou >= iou_threshold:
                    if det[idx2].prob<=obj.prob:
                        keep[idx2]=False
                    else:
                        is_keep=False
            det.append(obj)
            keep.append(is_keep)

        texts = []
        detections = []
        for idx in range(count):
            if not keep[idx]:
                continue

            # get detected face
            if args.face=="yolov3" or args.face=="yolov3-mask" or args.face=="mobilenet-mask":
                obj = org_detections[idx]
                margin = 1.0
                if args.face=="yolov3-mask" or args.face=="mobilenet-mask":
                    margin = 1.4
            else:
                obj = org_detections[idx]
                margin = 1.4

            cx = (obj.x + obj.w/2) * w
            cy = (obj.y + obj.h/2) * h
            cw = max(obj.w * w * margin,obj.h * h * margin)
            fx = max(cx - cw/2, 0)
            fy = max(cy - cw/2, 0)
            fw = min(cw, w-fx)
            fh = min(cw, h-fy)
            top_left = (int(fx), int(fy))
            bottom_right = (int((fx+fw)), int(fy+fh))

            # get detected face
            crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], 0:3]
            if crop_img.shape[0]<=0 or crop_img.shape[1]<=0:
                continue
            crop_img, resized_frame = adjust_frame_size(
                crop_img, IMAGE_HEIGHT, IMAGE_WIDTH
            )
            detections.append({"resized_frame":resized_frame,"top_left":top_left,"bottom_right":bottom_right,"id_sim":0,"score_sim":0,"fe":None})
        
        face_identification(tracks,net,detections,frame_no)
        frame_no=frame_no+1

        for detection in detections:
            # display result
            fontScale = w / 512.0
            thickness = 2
            color = hsv_to_rgb(256 * detection["id_sim"] / 16, 255, 255)
            cv2.rectangle(frame, detection["top_left"], detection["bottom_right"], color, 2)

            text_position = (int(detection["top_left"][0])+4, int((detection["bottom_right"][1])-8))

            cv2.putText(
                frame,
                f"{detection['id_sim']}",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                color,
                thickness
            )

        # display ui
        ui[:,0:w,:]=frame[:,:,:]
        ui[:,w:int(w*1.5),:]=0

        cnt = 0
        for i in range(len(tracks)):
            if len(tracks[i].image)<=0:
                continue

            y0=int(IMAGE_HEIGHT/4)*(cnt*2+0)
            y1=int(IMAGE_HEIGHT/4)*(cnt*2+1)
            y2=int(IMAGE_HEIGHT/4)*(cnt*2+2)

            for j in range(len(tracks[i].image)):
                x1=w+int(IMAGE_WIDTH/4)*j
                x2=w+int(IMAGE_WIDTH/4)*(j+1)
                if x2>=int(w*1.5) or y2>=h:
                    continue
                face=tracks[i].image[j]
                face=cv2.resize(face,((int)(IMAGE_WIDTH/4),(int)(IMAGE_HEIGHT/4)))
                ui[y1:y2,x1:x2,:]=face

            fontScale = 0.5
            thickness = 2
            color = hsv_to_rgb(256 * i / 16, 255, 255)
            cv2.rectangle(ui, (w,y0), (int(w*1.5),y2), color, 2)

            text_position = (w,y0 + 16)

            cv2.putText(
                ui,
                f"ID {i} : {tracks[i].score:5.3f}",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                color,
                thickness
            )

            cnt = cnt + 1


        #cv2.imshow('frame', frame)
        cv2.imshow('arcface', ui)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    if args.video:
        check_and_download_models(FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH)
    
    if args.video is None:
        # still image mode
        # comparing two images specified args.inputs
        compare_images()
    else:
        # video mode
        # comparing the specified image and the video
        compare_video()


if __name__ == "__main__":
    main()
