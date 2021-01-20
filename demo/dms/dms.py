import sys
import time
import argparse

import cv2
import numpy as np

import ailia

sys.path.append('../../hand_recognition/blazehand')
sys.path.append('../../face_recognition/facemesh')
sys.path.append('../../face_recognition/mediapipe_iris')
sys.path.append('../../pose_estimation/blazepose')

import blazehand_utils as bhut
import blazepose_utils as bput
import facemesh_utils as fut
import mediapipe_iris_utils as iut

sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# ======================
# Parameters 1
# ======================

IRIS_IMAGE_HEIGHT = 128
IRIS_IMAGE_WIDTH = 128

FACE_IMAGE_HEIGHT = 128
FACE_IMAGE_WIDTH = 128

HAND_IMAGE_HEIGHT = 256
HAND_IMAGE_WIDTH = 256

POSE_IMAGE_HEIGHT = 256
POSE_IMAGE_WIDTH = 256

# ======================
# Argument Parser Config
# ======================
SAVE_IMAGE_PATH="output.mp4"
parser = get_base_parser(
    'Driver monirogin system demo.', None, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
parser.add_argument(
    '-b', '--bbox',
    action='store_true',
    help='Display detected bonding box'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
FACE_LANDMARK2_MODEL_NAME = 'iris'
if args.normal:
    FACE_LANDMARK2_WEIGHT_PATH = f'{FACE_LANDMARK2_MODEL_NAME}.onnx'
    FACE_LANDMARK2_MODEL_PATH = f'{FACE_LANDMARK2_MODEL_NAME}.onnx.prototxt'
else:
    FACE_LANDMARK2_WEIGHT_PATH = f'{FACE_LANDMARK2_MODEL_NAME}.opt.onnx'
    FACE_LANDMARK2_MODEL_PATH = f'{FACE_LANDMARK2_MODEL_NAME}.opt.onnx.prototxt'
FACE_LANDMARK2_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/mediapipe_{FACE_LANDMARK2_MODEL_NAME}/'

FACE_DETECTION_MODEL_NAME = 'blazeface'
FACE_LANDMARK_MODEL_NAME = 'facemesh'
if args.normal:
    FACE_DETECTION_WEIGHT_PATH = f'{FACE_DETECTION_MODEL_NAME}.onnx'
    FACE_DETECTION_MODEL_PATH = f'{FACE_DETECTION_MODEL_NAME}.onnx.prototxt'
    FACE_LANDMARK_WEIGHT_PATH = f'{FACE_LANDMARK_MODEL_NAME}.onnx'
    FACE_LANDMARK_MODEL_PATH = f'{FACE_LANDMARK_MODEL_NAME}.onnx.prototxt'
else:
    FACE_DETECTION_WEIGHT_PATH = f'{FACE_DETECTION_MODEL_NAME}.opt.onnx'
    FACE_DETECTION_MODEL_PATH = f'{FACE_DETECTION_MODEL_NAME}.opt.onnx.prototxt'
    FACE_LANDMARK_WEIGHT_PATH = f'{FACE_LANDMARK_MODEL_NAME}.opt.onnx'
    FACE_LANDMARK_MODEL_PATH = f'{FACE_LANDMARK_MODEL_NAME}.opt.onnx.prototxt'
FACE_DETECTION_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{FACE_DETECTION_MODEL_NAME}/'
FACE_LANDMARK_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{FACE_LANDMARK_MODEL_NAME}/'

HAND_DETECTION_MODEL_NAME = 'blazepalm'
HAND_LANDMARK_MODEL_NAME = 'blazehand'
HAND_DETECTION_WEIGHT_PATH = f'{HAND_DETECTION_MODEL_NAME}.onnx'
HAND_DETECTION_MODEL_PATH = f'{HAND_DETECTION_MODEL_NAME}.onnx.prototxt'
HAND_LANDMARK_WEIGHT_PATH = f'{HAND_LANDMARK_MODEL_NAME}.onnx'
HAND_LANDMARK_MODEL_PATH = f'{HAND_LANDMARK_MODEL_NAME}.onnx.prototxt'
HAND_DETECTION_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{HAND_DETECTION_MODEL_NAME}/'
HAND_LANDMARK_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{HAND_LANDMARK_MODEL_NAME}/'

POSE_MODEL_NAME = 'blazepose'
POSE_DETECTOR_WEIGHT_PATH = f'{POSE_MODEL_NAME}_detector.onnx'
POSE_DETECTOR_MODEL_PATH = f'{POSE_MODEL_NAME}_detector.onnx.prototxt'
POSE_ESTIMATOR_WEIGHT_PATH = f'{POSE_MODEL_NAME}_estimator.onnx'
POSE_ESTIMATOR_MODEL_PATH = f'{POSE_MODEL_NAME}_estimator.onnx.prototxt'
POSE_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{POSE_MODEL_NAME}/'


# ======================
# Utils
# ======================
def draw_landmarks_hand(img, points, connections=[], color=(0, 0, 255), size=2):
    id = 0
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        color = hsv_to_rgb(255*id/bput.BLAZEPOSE_KEYPOINT_CNT, 255, 255)
        cv2.line(img, (x0, y0), (x1, y1), color, size)
        id = id + 1
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        #cv2.circle(img, (x, y), size+1, color, thickness=cv2.FILLED)

def draw_roi(img, roi):
    for i in range(roi.shape[0]):
        (x1,x2,x3,x4), (y1,y2,y3,y4) = roi[i]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0,255,0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0,0,0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0,0,0), 2)


def draw_landmarks_face(img, points, color=(0, 0, 255), size=2):
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, thickness=cv2.FILLED)


def draw_eye_iris(img, eyes, iris, eye_color=(0, 0, 255), iris_color=(255, 0, 0),
                  iris_pt_color=(0, 255, 0), size=1):
    """
    docstring
    """
    EYE_CONTOUR_ORDERED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 14, 13, 12, 11, 10, 9]

    for i in range(2):
        pts = eyes[i, EYE_CONTOUR_ORDERED, :2].round().astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, eye_color, thickness=size)

        center = tuple(iris[i, 0])
        radius = int(np.linalg.norm(iris[i, 1] - iris[i, 0]).round())
        cv2.circle(img, center, radius, iris_color, thickness=size)
        draw_landmarks_face(img, iris[i], color=iris_pt_color, size=size)


def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def line(input_img, landmarks, flags, point1, point2):
    threshold = 0.5
    for i in range(len(flags)):
        landmark, flag = landmarks[i], flags[i]
        if flag > threshold:
            color = hsv_to_rgb(255*point1/bput.BLAZEPOSE_KEYPOINT_CNT, 255, 255)

            x1 = int(landmark[point1, 0])
            y1 = int(landmark[point1, 1])
            x2 = int(landmark[point2, 0])
            y2 = int(landmark[point2, 1])
            cv2.line(input_img, (x1, y1), (x2, y2), color, 5)


def display_result_pose(input_img, count, landmarks, flags):
    for _ in range(count):
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_NOSE,bput.BLAZEPOSE_KEYPOINT_EYE_LEFT_INNER)
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_EYE_LEFT_INNER,bput.BLAZEPOSE_KEYPOINT_EYE_LEFT)
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_EYE_LEFT,bput.BLAZEPOSE_KEYPOINT_EYE_LEFT_OUTER)
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_EYE_LEFT_OUTER,bput.BLAZEPOSE_KEYPOINT_EAR_LEFT)

        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_NOSE,bput.BLAZEPOSE_KEYPOINT_EYE_RIGHT_INNER)
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_EYE_RIGHT_INNER,bput.BLAZEPOSE_KEYPOINT_EYE_RIGHT)
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_EYE_RIGHT,bput.BLAZEPOSE_KEYPOINT_EYE_RIGHT_OUTER)
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_EYE_RIGHT_OUTER,bput.BLAZEPOSE_KEYPOINT_EAR_RIGHT)

        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_MOUTH_LEFT,bput.BLAZEPOSE_KEYPOINT_MOUTH_RIGHT)

        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,bput.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,bput.BLAZEPOSE_KEYPOINT_ELBOW_LEFT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_ELBOW_LEFT,bput.BLAZEPOSE_KEYPOINT_WRIST_LEFT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT,bput.BLAZEPOSE_KEYPOINT_ELBOW_RIGHT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_ELBOW_RIGHT,bput.BLAZEPOSE_KEYPOINT_WRIST_RIGHT)

        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_WRIST_LEFT,bput.BLAZEPOSE_KEYPOINT_PINKY_LEFT_KNUCKLE1)
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_PINKY_LEFT_KNUCKLE1,bput.BLAZEPOSE_KEYPOINT_INDEX_LEFT_KNUCKLE1)
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_WRIST_LEFT,bput.BLAZEPOSE_KEYPOINT_INDEX_LEFT_KNUCKLE1)
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_WRIST_LEFT,bput.BLAZEPOSE_KEYPOINT_THUMB_LEFT_KNUCKLE2)

        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,bput.BLAZEPOSE_KEYPOINT_PINKY_RIGHT_KNUCKLE1)
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_PINKY_RIGHT_KNUCKLE1,bput.BLAZEPOSE_KEYPOINT_INDEX_RIGHT_KNUCKLE1)
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,bput.BLAZEPOSE_KEYPOINT_INDEX_RIGHT_KNUCKLE1)
        #line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,bput.BLAZEPOSE_KEYPOINT_THUMB_RIGHT_KNUCKLE2)

        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,bput.BLAZEPOSE_KEYPOINT_HIP_LEFT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT,bput.BLAZEPOSE_KEYPOINT_HIP_RIGHT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_HIP_LEFT,bput.BLAZEPOSE_KEYPOINT_HIP_RIGHT)

def display_hand_box(img, detections, with_keypoints=False):
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    n_keypoints = detections.shape[1] // 2 - 2

    for i in range(detections.shape[0]):
        ymin = detections[i, 0]
        xmin = detections[i, 1]
        ymax = detections[i, 2]
        xmax = detections[i, 3]
        
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2) 

        if with_keypoints:
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k*2    ])
                kp_y = int(detections[i, 4 + k*2 + 1])
                cv2.circle(img, (kp_x, kp_y), 2, (0, 0, 255), thickness=2)
    return img

# ======================
# Main functions
# ======================

def recognize_hand(frame,detector,estimator,out_frame=None):
    img256, _, scale, pad = bhut.resize_pad(frame[:,:,::-1])
    input_data = img256.astype('float32') / 255.
    input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

    # inference
    # Palm detection
    preds = detector.predict([input_data])
    detections = bhut.detector_postprocess(preds,anchor_path="../../hand_recognition/blazehand/anchors.npy")

    # display bbox
    if args.bbox:
        detections2 = bhut.denormalize_detections(detections[0].copy(), scale, pad)
        display_hand_box(out_frame, detections2)

    # Hand landmark estimation
    presence = [0, 0] # [left, right]
    if detections[0].size != 0:
        img, affine, _ = bhut.estimator_preprocess(frame, detections, scale, pad)
        estimator.set_input_shape(img.shape)
        flags, handedness, normalized_landmarks = estimator.predict([img])

        # postprocessing
        landmarks = bhut.denormalize_landmarks(normalized_landmarks, affine)
        for i in range(len(flags)):
            landmark, flag, handed = landmarks[i], flags[i], handedness[i]
            if flag > 0.75:
                if handed > 0.5:
                    presence[0] = 1
                else:
                    presence[1] = 1
                draw_landmarks_hand(out_frame, landmark[:,:2], bhut.HAND_CONNECTIONS, size=4)

    #if presence[0] and presence[1]:
    #    text = 'Left and right'
    #elif presence[0]:
    #    text = 'Left'
    #elif presence[1]:
    #    text = 'Right'
    #else:
    #    text = 'No hand'
    #cv2.putText(out_frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)



def recognize_pose(frame,detector,estimator,out_frame=None):
    _, img128, scale, pad = bput.resize_pad(frame[:,:,::-1])
    input_data = img128.astype('float32') / 255.
    input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

    # inference
    # Person detection
    detector_out = detector.predict([input_data])
    detections = bput.detector_postprocess(detector_out,anchor_path="../../pose_estimation/blazepose/anchors.npy",min_score_thresh = 0.5)
    count = len(detections) if detections[0].size > 0 else 0

    # display bbox
    if args.bbox:
        detections2 = bput.denormalize_detections(detections[0].copy(), scale, pad)
        display_hand_box(out_frame, detections2)

    # Pose estimation
    landmarks = []
    flags = []
    if count > 0:
        imgs, affine, _ = bput.estimator_preprocess(frame, detections, scale, pad)

        #flags, normalized_landmarks, _ = estimator.predict([imgs])
        #print(flags.shape)
        #print(normalized_landmarks.shape)

        flags = np.zeros((imgs.shape[0]))
        normalized_landmarks = np.zeros((imgs.shape[0], 31, 4))
        for i in range(imgs.shape[0]):
            flag, normalized_landmark, _ = estimator.predict([imgs[i:i+1,:,:,:]])
            flags[i]=flag
            normalized_landmarks[i]=normalized_landmark

        landmarks = bput.denormalize_landmarks(normalized_landmarks, affine)

    # postprocessing
    display_result_pose(out_frame, count, landmarks, flags)



def recognize_iris(frame,detector,estimator,estimator2,out_frame=None):
    _, img128, scale, pad = iut.resize_pad(frame[:,:,::-1])
    input_data = img128.astype('float32') / 127.5 - 1.0
    input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

    # inference
    # Face detection
    preds = detector.predict([input_data])
    detections = iut.detector_postprocess(preds,anchor_path="../../face_recognition/facemesh/anchors.npy")

    # display bbox
    if args.bbox:
        detections2 = iut.denormalize_detections(detections[0].copy(), scale, pad)
        display_hand_box(out_frame, detections2)

    # Face landmark estimation
    if detections[0].size != 0:
        imgs, affines, box = iut.estimator_preprocess(frame[:,:,::-1], detections, scale, pad)

        landmarks = np.zeros((imgs.shape[0], 1404))
        normalized_landmarks = np.zeros((imgs.shape[0], 468, 3))
        confidences = np.zeros((imgs.shape[0], 1))
        for i in range(imgs.shape[0]):
            landmark, confidences[i,:] = estimator.predict([imgs[i:i+1,:,:,:]])
            normalized_landmark = landmark / 192.0
            normalized_landmarks[i,:,:] = fut.denormalize_landmarks(normalized_landmark, affines)

            landmarks[i,:] = landmark

        #Added
        for i in range(len(normalized_landmark)):
            landmark, confidence = normalized_landmarks[i], confidences[i]
            draw_landmarks_face(out_frame, landmark[:,:2], size=1)

        # Iris landmark estimation
        imgs2, origins = iut.iris_preprocess(imgs, landmarks)

        eyes = np.zeros((imgs2.shape[0], 213))
        iris = np.zeros((imgs2.shape[0], 15))
        for i in range(imgs2.shape[0]):
            eyes[i,:], iris[i,:] = estimator2.predict([imgs2[i:i+1,:,:,:]])

        eyes, iris = iut.iris_postprocess(eyes, iris, origins, affines)
        for i in range(len(eyes)):
            draw_eye_iris(out_frame, eyes[i, :, :16, :2], iris[i, :, :, :2], size=2)

# ======================
# Main
# ======================


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    iris_detector = ailia.Net(FACE_DETECTION_MODEL_PATH, FACE_DETECTION_WEIGHT_PATH, env_id=env_id)
    iris_estimator = ailia.Net(FACE_LANDMARK_MODEL_PATH, FACE_LANDMARK_WEIGHT_PATH, env_id=env_id)
    iris_estimator2 = ailia.Net(FACE_LANDMARK2_MODEL_PATH, FACE_LANDMARK2_WEIGHT_PATH, env_id=env_id)

    hand_detector = ailia.Net(HAND_DETECTION_MODEL_PATH, HAND_DETECTION_WEIGHT_PATH, env_id=env_id)
    hand_estimator = ailia.Net(HAND_LANDMARK_MODEL_PATH, HAND_LANDMARK_WEIGHT_PATH, env_id=env_id)

    pose_detector = ailia.Net(POSE_DETECTOR_MODEL_PATH, POSE_DETECTOR_WEIGHT_PATH, env_id=env_id)
    pose_estimator = ailia.Net(POSE_ESTIMATOR_MODEL_PATH, POSE_ESTIMATOR_WEIGHT_PATH, env_id=env_id)

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        
        frame = np.ascontiguousarray(frame[:,::-1,:])
        out_frame = frame.copy()
        
        recognize_iris(frame,iris_detector,iris_estimator,iris_estimator2,out_frame=out_frame)
        recognize_pose(frame,pose_detector,pose_estimator,out_frame=out_frame)
        recognize_hand(frame,hand_detector,hand_estimator,out_frame=out_frame)

        cv2.imshow('frame', out_frame)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')
    pass




def main():
    # model files check and download
    check_and_download_models(FACE_DETECTION_WEIGHT_PATH, FACE_DETECTION_MODEL_PATH, FACE_DETECTION_REMOTE_PATH)
    check_and_download_models(FACE_LANDMARK_WEIGHT_PATH, FACE_LANDMARK_MODEL_PATH, FACE_LANDMARK_REMOTE_PATH)
    check_and_download_models(FACE_LANDMARK2_WEIGHT_PATH, FACE_LANDMARK2_MODEL_PATH, FACE_LANDMARK2_REMOTE_PATH)

    # model files check and download
    check_and_download_models(POSE_DETECTOR_WEIGHT_PATH, POSE_DETECTOR_MODEL_PATH, POSE_REMOTE_PATH)
    check_and_download_models(POSE_ESTIMATOR_WEIGHT_PATH, POSE_ESTIMATOR_MODEL_PATH, POSE_REMOTE_PATH)

    # model files check and download
    check_and_download_models(HAND_DETECTION_WEIGHT_PATH, HAND_DETECTION_MODEL_PATH, HAND_DETECTION_REMOTE_PATH)
    check_and_download_models(HAND_LANDMARK_WEIGHT_PATH, HAND_LANDMARK_MODEL_PATH, HAND_LANDMARK_REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        print("image mode is not supported.")


if __name__ == '__main__':
    main()
