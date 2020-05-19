import sys
import argparse

import numpy as np
import cv2

import ailia
from sort.tracker import Tracker
from sort.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils import *

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402


# ======================
# Parameters
# ======================
DT_WEIGHT_PATH = 'yolov3.opt.onnx'
DT_MODEL_PATH = 'yolov3.opt.onnx.prototxt'
EX_WEIGHT_PATH = 'deep_sort.onnx'
EX_MODEL_PATH = 'deep_sort.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deep_sort/'

VIDEO_PATH = 'sample.mp4'
SAVE_VIDEO_PATH = 'output.avi'

# Deep sort model input
INPUT_HEIGHT = 128
INPUT_WIDTH = 64

# yolo params
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
THRESHOLD = 0.1  # 0.4
MIN_CONFIDENCE = 0.3
IOU = 0.45


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Deep SORT'
)
parser.add_argument(
    '-i', '--input', metavar='VIDEO',
    default=VIDEO_PATH,
    help=('The input video path.' +
          'If the VIDEO argument is set to 0, the webcam input will be used.')
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_VIDEO_PATH',
    default=SAVE_VIDEO_PATH,
    help='Save path for the output video.'
)
args = parser.parse_args()


# ======================
# Utils
# ======================
def resize(img, size=(INPUT_WIDTH, INPUT_HEIGHT)):
    return cv2.resize(img.astype(np.float32), size)


# ======================
# Main functions
# ======================
def recognize_from_video():
    results = []
    idx_frame = 0

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')

    # Detector
    detector = ailia.Detector(
        DT_MODEL_PATH,
        DT_WEIGHT_PATH,
        len(COCO_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=env_id
    )

    # SORT Model
    extractor = ailia.Net(EX_MODEL_PATH, EX_WEIGHT_PATH, env_id=env_id)

    # tracker class instance
    max_cosine_distance = 0.2
    nn_budget = 100
    metric = NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(
        metric,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3
    )

    if args.input == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    else:
        if check_file_existance(args.input):
            capture = cv2.VideoCapture(args.input)

    # create video writer
    writer = cv2.VideoWriter(
        args.savepath,
        # cv2.VideoWriter_fourcc(*'mpeg'),
        cv2.VideoWriter_fourcc(*'MJPG'),  # mp4
        # cv2.VideoWriter_fourcc(*'XVID'),  # avi
        20,
        (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    )

    print('Start Inference...')
    while(True):
        idx_frame += 1
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            break

        input_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
        h, w = frame.shape[0], frame.shape[1]

        # do detection
        detector.compute(input_img, THRESHOLD, IOU)
        bbox_xywh, cls_conf, cls_ids = get_detector_result(detector, h, w)
        
        # select person class
        mask = cls_ids == 0
        bbox_xywh = bbox_xywh[mask]
        # bbox dilation just in case bbox too small,
        # delete this line if using a better pedestrian detector
        bbox_xywh[:, 3:] *= 1.2
        cls_conf = cls_conf[mask]

        # do tracking
        img_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
            img_crops.append(frame[y1:y2, x1:x2])

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

        # draw box for visualization
        if len(outputs) > 0:
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            frame = draw_boxes(frame, bbox_xyxy, identities)

            for bb_xyxy in bbox_xyxy:
                bbox_tlwh.append(xyxy_to_tlwh(bb_xyxy))

            results.append((idx_frame - 1, bbox_tlwh, identities))

        cv2.imshow('frame', frame)

        if args.savepath:
            writer.write(frame)

        write_results(args.savepath.split('.')[0] + '.txt', results, 'mot')

    capture.release()
    cv2.destroyAllWindows()
    print(f'Save results to {args.savepath}')
    print('Script finished successfully.')


def main():
    # model files check and download
    print('Check Detector...')
    check_and_download_models(DT_WEIGHT_PATH, DT_MODEL_PATH, REMOTE_PATH)
    print('Check Extractor...')
    check_and_download_models(EX_WEIGHT_PATH, EX_MODEL_PATH, REMOTE_PATH)

    recognize_from_video()


if __name__ == '__main__':
    main()
