import sys
import time
import argparse

import cv2
import numpy as np

import ailia

sys.path.append('../util')
from webcamera_utils import adjust_frame_size  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import check_file_existance  # noqa: E402

# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'balloon.png'
SAVE_IMAGE_PATH = '???'
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

ALGORITHM = ailia.POSE_ALGORITHM_LW_HUMAN_POSE


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Fast and accurate human pose 2D-estimation.'
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
    '-n', '--normal',
    action='store_false',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
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


# ======================
# Utils
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def line(input_img, person, point1, point2):
    threshold = 0.2
    if person.points[point1].score > threshold and\
       person.points[point2].score > threshold:
        color = hsv_to_rgb(255*point1/ailia.POSE_KEYPOINT_CNT, 255, 255)

        x1 = int(input_img.shape[1] * person.points[point1].x)
        y1 = int(input_img.shape[0] * person.points[point1].y)
        x2 = int(input_img.shape[1] * person.points[point2].x)
        y2 = int(input_img.shape[0] * person.points[point2].y)
        cv2.line(input_img, (x1, y1), (x2, y2), color, 5)


def display_result(input_img, pose):
    count = pose.get_object_count()
    for idx in range(count):
        person = pose.get_object_pose(idx)
        for i in range(ailia.POSE_KEYPOINT_CNT):
            # score = person.points[i].score
            # x = (input_img.shape[1] * person.points[i].x)
            # y = (input_img.shape[0] * person.points[i].y)

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


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    src_img = cv2.imread(args.input)
    input_image = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None'
    )
    input_data = cv2.cvtColor(input_image, cv2.COLOR_BGR2BGRA)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    pose = ailia.PoseEstimator(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id, algorithm=ALGORITHM
    )

    # compute execution time
    for i in range(5):
        start = int(round(time.time() * 1000))
        _ = pose.compute(input_data)
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')

    # postprocessing
    count = pose.get_object_count()
    print(f'person_count={count}')
    display_result(src_img, pose)
    cv2.imwrite(args.savepath, src_img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    pose = ailia.PoseEstimator(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id, algorithm=ALGORITHM
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

    while(True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        input_image, input_data = adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH,
        )
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2BGRA)

        # inferece
        _ = pose.compute(input_data)

        # postprocessing
        display_result(input_image, pose)
        cv2.imshow('frame', input_image)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
