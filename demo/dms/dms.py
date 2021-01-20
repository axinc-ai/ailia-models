import sys
import time
import argparse

import cv2
import numpy as np

import ailia
import blazehand_utils as bhut
import blazepose_utils as bput
import facemesh_utils as fut
import mediapipe_iris_utils as iut

sys.path.append('../../util')
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
parser = get_base_parser(
    'Driver monirogin system demo.', None, None,
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
FACE_LANDMARK2_MODEL_NAME = 'iris'
if args.normal:
    FACE_LANDMARK2_WEIGHT_PATH = f'{LANDMARK2_MODEL_NAME}.onnx'
    FACE_LANDMARK2_MODEL_PATH = f'{LANDMARK2_MODEL_NAME}.onnx.prototxt'
else:
    FACE_LANDMARK2_WEIGHT_PATH = f'{LANDMARK2_MODEL_NAME}.opt.onnx'
    FACE_LANDMARK2_MODEL_PATH = f'{LANDMARK2_MODEL_NAME}.opt.onnx.prototxt'
FACE_LANDMARK2_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/mediapipe_{LANDMARK2_MODEL_NAME}/'

FACE_DETECTION_MODEL_NAME = 'blazeface'
FACE_LANDMARK_MODEL_NAME = 'facemesh'
if args.normal:
    FACE_DETECTION_WEIGHT_PATH = f'{DETECTION_MODEL_NAME}.onnx'
    FACE_DETECTION_MODEL_PATH = f'{DETECTION_MODEL_NAME}.onnx.prototxt'
    FACE_LANDMARK_WEIGHT_PATH = f'{LANDMARK_MODEL_NAME}.onnx'
    FACE_LANDMARK_MODEL_PATH = f'{LANDMARK_MODEL_NAME}.onnx.prototxt'
else:
    FACE_DETECTION_WEIGHT_PATH = f'{DETECTION_MODEL_NAME}.opt.onnx'
    FACE_DETECTION_MODEL_PATH = f'{DETECTION_MODEL_NAME}.opt.onnx.prototxt'
    FACE_LANDMARK_WEIGHT_PATH = f'{LANDMARK_MODEL_NAME}.opt.onnx'
    FACE_LANDMARK_MODEL_PATH = f'{LANDMARK_MODEL_NAME}.opt.onnx.prototxt'
FACE_DETECTION_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{DETECTION_MODEL_NAME}/'
FACE_LANDMARK_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{LANDMARK_MODEL_NAME}/'

DETECTION_MODEL_NAME = 'blazepalm'
LANDMARK_MODEL_NAME = 'blazehand'
HAND_DETECTION_WEIGHT_PATH = f'{DETECTION_MODEL_NAME}.onnx'
HAND_DETECTION_MODEL_PATH = f'{DETECTION_MODEL_NAME}.onnx.prototxt'
HAND_LANDMARK_WEIGHT_PATH = f'{LANDMARK_MODEL_NAME}.onnx'
HAND_LANDMARK_MODEL_PATH = f'{LANDMARK_MODEL_NAME}.onnx.prototxt'
HAND_DETECTION_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{DETECTION_MODEL_NAME}/'
HAND_LANDMARK_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{LANDMARK_MODEL_NAME}/'

MODEL_NAME = 'blazepose'
POSE_DETECTOR_WEIGHT_PATH = f'{MODEL_NAME}_detector.onnx'
POSE_DETECTOR_MODEL_PATH = f'{MODEL_NAME}_detector.onnx.prototxt'
POSE_ESTIMATOR_WEIGHT_PATH = f'{MODEL_NAME}_estimator.onnx'
POSE_ESTIMATOR_MODEL_PATH = f'{MODEL_NAME}_estimator.onnx.prototxt'
POSE_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'


# ======================
# Utils
# ======================
def draw_landmarks(img, points, connections=[], color=(0, 0, 255), size=2):
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), size)
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size+1, color, thickness=cv2.FILLED)


# ======================
# Main functions
# ======================


def recognize_from_video_hand():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Net(HAND_DETECTION_MODEL_PATH, HAND_DETECTION_WEIGHT_PATH, env_id=env_id)
    estimator = ailia.Net(HAND_LANDMARK_MODEL_PATH, HAND_LANDMARK_WEIGHT_PATH, env_id=env_id)

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
        frame = np.ascontiguousarray(frame[:,::-1,:])
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        img256, _, scale, pad = bhut.resize_pad(frame[:,:,::-1])
        input_data = img256.astype('float32') / 255.
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        # Palm detection
        preds = detector.predict([input_data])
        detections = bhut.detector_postprocess(preds,anchors="../../hand_recognition/blazehand/anchors.npy")

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
                    draw_landmarks(frame, landmark[:,:2], bhut.HAND_CONNECTIONS, size=2)

        if presence[0] and presence[1]:
            text = 'Left and right'
        elif presence[0]:
            text = 'Left'
        elif presence[1]:
            text = 'Right'
        else:
            text = 'No hand'
        cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow('frame', frame)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')
    pass




# ======================
# Utils
# ======================
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


def display_result(input_img, count, landmarks, flags):
    for _ in range(count):
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_NOSE,bput.BLAZEPOSE_KEYPOINT_EYE_LEFT_INNER)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_EYE_LEFT_INNER,bput.BLAZEPOSE_KEYPOINT_EYE_LEFT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_EYE_LEFT,bput.BLAZEPOSE_KEYPOINT_EYE_LEFT_OUTER)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_EYE_LEFT_OUTER,bput.BLAZEPOSE_KEYPOINT_EAR_LEFT)

        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_NOSE,bput.BLAZEPOSE_KEYPOINT_EYE_RIGHT_INNER)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_EYE_RIGHT_INNER,bput.BLAZEPOSE_KEYPOINT_EYE_RIGHT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_EYE_RIGHT,bput.BLAZEPOSE_KEYPOINT_EYE_RIGHT_OUTER)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_EYE_RIGHT_OUTER,bput.BLAZEPOSE_KEYPOINT_EAR_RIGHT)

        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_MOUTH_LEFT,bput.BLAZEPOSE_KEYPOINT_MOUTH_RIGHT)

        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,bput.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,bput.BLAZEPOSE_KEYPOINT_ELBOW_LEFT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_ELBOW_LEFT,bput.BLAZEPOSE_KEYPOINT_WRIST_LEFT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT,bput.BLAZEPOSE_KEYPOINT_ELBOW_RIGHT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_ELBOW_RIGHT,bput.BLAZEPOSE_KEYPOINT_WRIST_RIGHT)

        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_WRIST_LEFT,bput.BLAZEPOSE_KEYPOINT_PINKY_LEFT_KNUCKLE1)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_PINKY_LEFT_KNUCKLE1,bput.BLAZEPOSE_KEYPOINT_INDEX_LEFT_KNUCKLE1)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_WRIST_LEFT,bput.BLAZEPOSE_KEYPOINT_INDEX_LEFT_KNUCKLE1)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_WRIST_LEFT,bput.BLAZEPOSE_KEYPOINT_THUMB_LEFT_KNUCKLE2)

        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,bput.BLAZEPOSE_KEYPOINT_PINKY_RIGHT_KNUCKLE1)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_PINKY_RIGHT_KNUCKLE1,bput.BLAZEPOSE_KEYPOINT_INDEX_RIGHT_KNUCKLE1)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,bput.BLAZEPOSE_KEYPOINT_INDEX_RIGHT_KNUCKLE1)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,bput.BLAZEPOSE_KEYPOINT_THUMB_RIGHT_KNUCKLE2)

        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,bput.BLAZEPOSE_KEYPOINT_HIP_LEFT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT,bput.BLAZEPOSE_KEYPOINT_HIP_RIGHT)
        line(input_img,landmarks,flags,bput.BLAZEPOSE_KEYPOINT_HIP_LEFT,bput.BLAZEPOSE_KEYPOINT_HIP_RIGHT)




# ======================
# Main functions
# ======================



def recognize_from_video_pose():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Net(POSE_DETECTOR_MODEL_PATH, POSE_DETECTOR_WEIGHT_PATH, env_id=env_id)
    estimator = ailia.Net(POSE_ESTIMATOR_MODEL_PATH, POSE_ESTIMATOR_WEIGHT_PATH, env_id=env_id)

    capture = get_capture(args.video)

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        _, img128, scale, pad = bput.resize_pad(frame[:,:,::-1])
        input_data = img128.astype('float32') / 255.
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        # Person detection
        detector_out = detector.predict([input_data])
        detections = bput.detector_postprocess(detector_out,anchors="../../pose_estimatinon/blazepose/anchors.npy")
        count = len(detections) if detections[0].size > 0 else 0

        # Pose estimation
        landmarks = []
        flags = []
        if count > 0:
            img, affine, _ = bput.estimator_preprocess(frame, detections, scale, pad)
            flags, normalized_landmarks, _ = estimator.predict([img])
            landmarks = bput.denormalize_landmarks(normalized_landmarks, affine)

        # postprocessing
        display_result(frame, count, landmarks, flags)
        cv2.imshow('frame', frame)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')
    pass

# ======================
# Utils
# ======================
def draw_roi(img, roi):
    for i in range(roi.shape[0]):
        (x1,x2,x3,x4), (y1,y2,y3,y4) = roi[i]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0,255,0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0,0,0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0,0,0), 2)


def draw_landmarks(img, points, color=(0, 0, 255), size=2):
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, thickness=cv2.FILLED)


# ======================
# Main functions
# ======================



def recognize_from_video_face():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Net(FACE_DETECTION_MODEL_PATH, FACE_DETECTION_WEIGHT_PATH, env_id=env_id)
    estimator = ailia.Net(FACE_LANDMARK_MODEL_PATH, FACE_LANDMARK_WEIGHT_PATH, env_id=env_id)

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

        _, img128, scale, pad = fut.resize_pad(frame[:,:,::-1])
        input_data = img128.astype('float32') / 127.5 - 1.0
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        # Face detection
        preds = detector.predict([input_data])
        detections = fut.detector_postprocess(preds,anchors="../../face_recognition/facemesh/anchors.npy")

        # Face landmark estimation
        if detections[0].size != 0:
            imgs, affines, box = fut.estimator_preprocess(frame[:,:,::-1], detections, scale, pad)
            draw_roi(frame, box)

            dynamic_input_shape = False

            if dynamic_input_shape:
                estimator.set_input_shape(imgs.shape)
                landmarks, confidences = estimator.predict([imgs])
                normalized_landmarks = landmarks / 192.0
                landmarks = fut.denormalize_landmarks(normalized_landmarks, affines)
            else:
                landmarks = np.zeros((imgs.shape[0], 468, 3))
                confidences = np.zeros((imgs.shape[0], 1))
                for i in range(imgs.shape[0]):
                    landmark, confidences[i,:] = estimator.predict([imgs[i:i+1,:,:,:]])
                    normalized_landmark = landmark / 192.0
                    landmarks[i,:,:] = fut.denormalize_landmarks(normalized_landmark, affines)

            for i in range(len(landmarks)):
                landmark, confidence = landmarks[i], confidences[i]
                # if confidence > 0: # Can be > 1, no idea what it represents
                draw_landmarks(frame, landmark[:,:2], size=1)

        cv2.imshow('frame', frame)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')
    pass




# ======================
# Utils
# ======================
def draw_roi(img, roi):
    for i in range(roi.shape[0]):
        (x1,x2,x3,x4), (y1,y2,y3,y4) = roi[i]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0,255,0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0,0,0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0,0,0), 2)


def draw_landmarks(img, points, color=(0, 0, 255), size=2):
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
        draw_landmarks(img, iris[i], color=iris_pt_color, size=size)

# ======================
# Main functions
# ======================



def recognize_from_video_iris():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Net(FACE_DETECTION_MODEL_PATH, FACE_DETECTION_WEIGHT_PATH, env_id=env_id)
    estimator = ailia.Net(FACE_LANDMARK_MODEL_PATH, FACE_LANDMARK_WEIGHT_PATH, env_id=env_id)
    estimator2 = ailia.Net(FACE_LANDMARK2_MODEL_PATH, FACE_LANDMARK2_WEIGHT_PATH, env_id=env_id)

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

        _, img128, scale, pad = iut.resize_pad(frame[:,:,::-1])
        input_data = img128.astype('float32') / 127.5 - 1.0
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        # Face detection
        preds = detector.predict([input_data])
        detections = iut.detector_postprocess(preds,anchors="../../face_recognition/facemesh/anchors.npy")

        # Face landmark estimation
        if detections[0].size != 0:
            imgs, affines, box = iut.estimator_preprocess(frame[:,:,::-1], detections, scale, pad)

            dynamic_shape = False

            if dynamic_shape:
                estimator.set_input_shape(imgs.shape)
                landmarks, confidences = estimator.predict([imgs])
            else:
                landmarks = np.zeros((imgs.shape[0], 1404))
                confidences = np.zeros((imgs.shape[0], 1))
                for i in range(imgs.shape[0]):
                    landmarks[i,:], confidences[i,:] = estimator.predict([imgs[i:i+1,:,:,:]])

            # Iris landmark estimation
            imgs2, origins = iut.iris_preprocess(imgs, landmarks)

            if dynamic_shape:
                estimator2.set_input_shape(imgs2.shape)
                eyes, iris = estimator2.predict([imgs2])
            else:
                eyes = np.zeros((imgs2.shape[0], 213))
                iris = np.zeros((imgs2.shape[0], 15))
                for i in range(imgs2.shape[0]):
                    eyes[i,:], iris[i,:] = estimator2.predict([imgs2[i:i+1,:,:,:]])

            eyes, iris = iut.iris_postprocess(eyes, iris, origins, affines)
            for i in range(len(eyes)):
                draw_eye_iris(frame, eyes[i, :, :16, :2], iris[i, :, :, :2], size=1)

        cv2.imshow('frame', frame)

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
        recognize_from_video_iris()
        recognize_from_video_face()
        recognize_from_video_pose()
        recognize_from_video_hand()
    else:
        # image mode
        print("image mode is not supported.")


if __name__ == '__main__':
    main()
