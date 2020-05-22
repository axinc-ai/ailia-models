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
from detector_utils import plot_results, load_image  # noqa: E402C
from webcamera_utils import get_writer  # noqa: E402C


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

# Metric parameters
MAX_COSINE_DISTANCE = 0.2  # threshold of matching object
NN_BUDGET = 100


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Deep SORT'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=VIDEO_PATH,
    help=('The input video path.'
          'If the VIDEO argument is set to 0, the webcam input will be used.')
)
parser.add_argument(
    '-p', '--pairimage', metavar='IMAGE',
    nargs=2,
    default=[None, None],
    help=('If this option is specified, the model is set to determine '
          'if the person in two images is the same person or not.')
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


def init_detector(env_id):
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
    return detector


# ======================
# Main functions
# ======================
def recognize_from_video():
    results = []
    idx_frame = 0

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = init_detector(env_id)
    extractor = ailia.Net(EX_MODEL_PATH, EX_WEIGHT_PATH, env_id=env_id)

    # tracker class instance
    metric = NearestNeighborDistanceMetric(
        "cosine", MAX_COSINE_DISTANCE, NN_BUDGET
    )
    tracker = Tracker(
        metric,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3
    )

    if args.video == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    else:
        if check_file_existance(args.video):
            capture = cv2.VideoCapture(args.video)

    # create video writer
    writer = get_writer(
        args.savepath,
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    print('Start Inference...')
    while(True):
        idx_frame += 1
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # In order to use ailia.Detector, the input should have 4 channels.
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


def compare_images():
    """
    This is a mode to determine if two input images have the same person
    by using the CNN model, which is used in DeepSORT to track the same person.
    It is assumed that there is always only one person in each image.
    We have not verified, and do not assume, the behavor in the case of
    multiple people. (Future work)
    """

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = init_detector(env_id)
    extractor = ailia.Net(EX_MODEL_PATH, EX_WEIGHT_PATH, env_id=env_id)

    # prepare input data
    input_data = []
    for i in range(len(args.pairimage)):
        input_data.append(load_image(args.pairimage[i]))

    # inference
    print('Start inference...')
    features = []
    for i in range(len(input_data)):
        # do detection
        detector.compute(input_data[i], THRESHOLD, IOU)
        h, w = input_data[i].shape[0], input_data[i].shape[1]
        bbox_xywh, cls_conf, cls_ids = get_detector_result(detector, h, w)
        
        # select person class
        mask = cls_ids == 0
        if mask.sum() == 0:
            print('Detector could not detect any person '
                  f'in the input image: {args.pairimage[i]}')
            print('Program finished.')
            sys.exit(0)
        
        bbox_xywh = bbox_xywh[mask]

        # bbox dilation just in case bbox too small,
        # delete this line if using a better pedestrian detector
        bbox_xywh[:, 3:] *= 1.2
        cls_conf = cls_conf[mask]

        # image crop
        """
        [INFO] If more than one bounding box is detected,
        the one with the highest confidence is used as correct box.
        It should be noted that this works because we assume that
        the input image has only one person.
        """
        x1, y1, x2, y2 = xywh_to_xyxy(bbox_xywh[np.argmax(cls_conf)], h, w)
        src_img = cv2.cvtColor(input_data[i], cv2.COLOR_BGRA2RGB)
        img_crop = src_img[y1:y2, x1:x2]
            
        # preprocess
        img_crop = normalize_image(
            resize(img_crop), 'ImageNet'
        )[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        features.append(extractor.predict(img_crop)[0])

    sim = cosin_metric(features[0], features[1])
    if sim >= (1 - MAX_COSINE_DISTANCE):
        print(f'{args.pairimage}: SAME person (confidence: {sim})')
    else:
        print(f'{args.pairimage}: Diefferent person (confidence: {sim})')


def main():
    # model files check and download
    print('Check Detector...')
    check_and_download_models(DT_WEIGHT_PATH, DT_MODEL_PATH, REMOTE_PATH)
    print('Check Extractor...')
    check_and_download_models(EX_WEIGHT_PATH, EX_MODEL_PATH, REMOTE_PATH)

    if args.pairimage[0] != None and args.pairimage[1] != None:
        print('Cheking if the person in two images is the same person or not.')
        compare_images()
    else:
        print('Deep SORT started')
        recognize_from_video()


if __name__ == '__main__':
    main()
