import sys
import time
import argparse

import cv2
import numpy as np

import ailia

sys.path.append('../../util')
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'balloon.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


# ======================
# Argument Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='BlazePose, an on-device real-time body pose tracking.'
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
# parser.add_argument(
#     '-n', '--normal',
#     action='store_false',
#     help='By default, the optimized model is used, but with this option, ' +
#     'you can switch to the normal (not optimized) model'
# )
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
# parser.add_argument(
#     '-b', '--benchmark',
#     action='store_true',
#     help='Running the inference on the same input 5 times ' +
#          'to measure execution performance. (Cannot be used in video mode)'
# )
args = parser.parse_args()


# ======================
# Parameters 2
# ======================
MODEL_NAME = 'blazepose'
# if args.normal:
WEIGHT_PATH = f'{MODEL_NAME}.onnx'
MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'
# else:
    # WEIGHT_PATH = f'{MODEL_NAME}.opt.onnx'
    # MODEL_PATH = f'{MODEL_NAME}.opt.onnx.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'


# ======================
# Utils
# ======================
# def hsv_to_rgb(h, s, v):
#     bgr = cv2.cvtColor(
#         np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
#     )[0][0]
#     return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


# def line(input_img, person, point1, point2):
#     threshold = 0.3
#     if person.points[point1].score > threshold and\
#        person.points[point2].score > threshold:
#         color = hsv_to_rgb(255*point1/ailia.BLAZEPOSE_KEYPOINT_CNT, 255, 255)

#         x1 = int(input_img.shape[1] * person.points[point1].x)
#         y1 = int(input_img.shape[0] * person.points[point1].y)
#         x2 = int(input_img.shape[1] * person.points[point2].x)
#         y2 = int(input_img.shape[0] * person.points[point2].y)
#         cv2.line(input_img, (x1, y1), (x2, y2), color, 5)


# def display_result(input_img, pose):
#     count = pose.get_object_count()
#     for idx in range(count):
#         person = pose.get_object_pose(idx)

#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_NOSE,ailia.BLAZEPOSE_KEYPOINT_EYE_LEFT_INNER)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_EYE_LEFT_INNER,ailia.BLAZEPOSE_KEYPOINT_EYE_LEFT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_EYE_LEFT,ailia.BLAZEPOSE_KEYPOINT_EYE_LEFT_OUTER)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_EYE_LEFT_OUTER,ailia.BLAZEPOSE_KEYPOINT_EAR_LEFT)

#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_NOSE,ailia.BLAZEPOSE_KEYPOINT_EYE_RIGHT_INNER)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_EYE_RIGHT_INNER,ailia.BLAZEPOSE_KEYPOINT_EYE_RIGHT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_EYE_RIGHT,ailia.BLAZEPOSE_KEYPOINT_EYE_RIGHT_OUTER)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_EYE_RIGHT_OUTER,ailia.BLAZEPOSE_KEYPOINT_EAR_RIGHT)

#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_MOUTH_LEFT,ailia.BLAZEPOSE_KEYPOINT_MOUTH_RIGHT)

#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,ailia.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,ailia.BLAZEPOSE_KEYPOINT_ELBOW_LEFT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_ELBOW_LEFT,ailia.BLAZEPOSE_KEYPOINT_WRIST_LEFT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT,ailia.BLAZEPOSE_KEYPOINT_ELBOW_RIGHT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_ELBOW_RIGHT,ailia.BLAZEPOSE_KEYPOINT_WRIST_RIGHT)
        
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_WRIST_LEFT,ailia.BLAZEPOSE_KEYPOINT_PINKY_LEFT_KNUCKLE1)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_PINKY_LEFT_KNUCKLE1,ailia.BLAZEPOSE_KEYPOINT_INDEX_LEFT_KNUCKLE1)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_WRIST_LEFT,ailia.BLAZEPOSE_KEYPOINT_INDEX_LEFT_KNUCKLE1)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_WRIST_LEFT,ailia.BLAZEPOSE_KEYPOINT_THUMB_LEFT_KNUCKLE2)

#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,ailia.BLAZEPOSE_KEYPOINT_PINKY_RIGHT_KNUCKLE1)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_PINKY_RIGHT_KNUCKLE1,ailia.BLAZEPOSE_KEYPOINT_INDEX_RIGHT_KNUCKLE1)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,ailia.BLAZEPOSE_KEYPOINT_INDEX_RIGHT_KNUCKLE1)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,ailia.BLAZEPOSE_KEYPOINT_THUMB_RIGHT_KNUCKLE2)

#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,ailia.BLAZEPOSE_KEYPOINT_HIP_LEFT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT,ailia.BLAZEPOSE_KEYPOINT_HIP_RIGHT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_HIP_LEFT,ailia.BLAZEPOSE_KEYPOINT_HIP_RIGHT)

#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_HIP_LEFT,ailia.BLAZEPOSE_KEYPOINT_KNEE_LEFT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_KNEE_LEFT,ailia.BLAZEPOSE_KEYPOINT_ANKLE_LEFT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_HIP_RIGHT,ailia.BLAZEPOSE_KEYPOINT_KNEE_RIGHT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_KNEE_RIGHT,ailia.BLAZEPOSE_KEYPOINT_ANKLE_RIGHT)

#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_ANKLE_LEFT,ailia.BLAZEPOSE_KEYPOINT_HEEL_LEFT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_HEEL_LEFT,ailia.BLAZEPOSE_KEYPOINT_FOOT_LEFT_INDEX)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_ANKLE_LEFT,ailia.BLAZEPOSE_KEYPOINT_FOOT_LEFT_INDEX)

#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_ANKLE_RIGHT,ailia.BLAZEPOSE_KEYPOINT_HEEL_RIGHT)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_HEEL_RIGHT,ailia.BLAZEPOSE_KEYPOINT_FOOT_RIGHT_INDEX)
#         line(input_img,person,ailia.BLAZEPOSE_KEYPOINT_ANKLE_RIGHT,ailia.BLAZEPOSE_KEYPOINT_FOOT_RIGHT_INDEX)


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
    input_data = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGRA)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # inference
    print('Start inference...')
    # if args.benchmark:
    #     print('BENCHMARK mode')
    #     for i in range(5):
    #         start = int(round(time.time() * 1000))
    #         _ = pose.compute(input_data)
    #         end = int(round(time.time() * 1000))
    #         print(f'\tailia processing time {end - start} ms')
    # else:
    #     _ = pose.compute(input_data)
    preds_ailia = net.predict([input_data])

    # postprocessing
    # count = pose.get_object_count()
    # print(f'person_count={count}')
    # display_result(src_img, pose)
    # cv2.imwrite(args.savepath, src_img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    # env_id = ailia.get_gpu_environment_id()
    # print(f'env_id: {env_id}')
    # pose = ailia.PoseEstimator(
    #     MODEL_PATH, WEIGHT_PATH, env_id=env_id, algorithm=ALGORITHM
    # )

    # capture = get_capture(args.video)

    # while(True):
    #     ret, frame = capture.read()
    #     if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
    #         break

    #     input_image, input_data = adjust_frame_size(
    #         frame, IMAGE_HEIGHT, IMAGE_WIDTH,
    #     )
    #     input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2BGRA)

    #     # inferece
    #     _ = pose.compute(input_data)

    #     # postprocessing
    #     display_result(input_image, pose)
    #     cv2.imshow('frame', input_image)

    # capture.release()
    # cv2.destroyAllWindows()
    # print('Script finished successfully.')
    pass


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
