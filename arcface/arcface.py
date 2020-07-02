import sys
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../util')
from webcamera_utils import adjust_frame_size  # noqa: E402
from image_utils import load_image, draw_result_on_img  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import check_file_existance  # noqa: E402
from detector_utils import hsv_to_rgb # noqa: E402C


# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = 'arcface.onnx'
MODEL_PATH = 'arcface.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/arcface/"

YOLOV3_FACE_WEIGHT_PATH = 'yolov3-face.opt.onnx'
YOLOV3_FACE_MODEL_PATH = 'yolov3-face.opt.onnx.prototxt'
YOLOV3_FACE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3-face/'
YOLOV3_FACE_THRESHOLD = 0.2
YOLOV3_FACE_IOU = 0.45

IMG_PATH_1 = 'correct_pair_1.jpg'
IMG_PATH_2 = 'correct_pair_2.jpg'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# (IMAGE_HEIGHT * 2 * WEBCAM_SCALE, IMAGE_WIDTH * 2 * WEBCAM_SCALE)
# Scale to determine the input size of the webcam
WEBCAM_SCALE = 1.5

# the threshold was calculated by the `test_performance` function in `test.py`
# of the original repository
THRESHOLD = 0.25572845

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
args = parser.parse_args()


# ======================
# Utils
# ======================
def preprocess_image(image, input_is_bgr=False):
    # (ref: https://github.com/ronghuaiyang/arcface-pytorch/issues/14)
    # use origin image and fliped image to infer,
    # and concat the feature as the final feature of the origin image.
    if input_is_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(imgs)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = net.predict(imgs)

    # postprocessing
    fe_1 = np.concatenate([preds_ailia[0], preds_ailia[1]], axis=0)
    fe_2 = np.concatenate([preds_ailia[2], preds_ailia[3]], axis=0)
    sim = cosin_metric(fe_1, fe_2)

    print(f'Similarity of ({args.inputs[0]}, {args.inputs[1]}) : {sim:.3f}')
    if THRESHOLD > sim:
        print('They are not the same face!')
    else:
        print('They are the same face!')


def compare_image_and_video():
    # prepare base image
    fe_list = []

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # detector initialize
    detector = ailia.Detector(
        YOLOV3_FACE_MODEL_PATH,
        YOLOV3_FACE_WEIGHT_PATH,
        1,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=env_id
    )

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

    # inference loop
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # detect face
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        detector.compute(img, YOLOV3_FACE_THRESHOLD, YOLOV3_FACE_IOU)
        h, w = img.shape[0], img.shape[1]
        count = detector.get_object_count()
        texts = []
        for idx in range(count):
            # get detected face
            obj = detector.get_object(idx)
            cx = (obj.x + obj.w/2) * w
            cy = (obj.y + obj.h/2) * h
            margin = 1.0
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

            # prepare target face and input face
            input_frame = preprocess_image(resized_frame, input_is_bgr=True)
            input_data = np.concatenate([input_frame, input_frame], axis=0)

            # inference
            preds_ailia = net.predict(input_data)

            # postprocessing
            fe_1 = np.concatenate([preds_ailia[0], preds_ailia[1]], axis=0)
            fe_2 = np.concatenate([preds_ailia[2], preds_ailia[3]], axis=0)

            # get matched face
            id_sim = 0
            score_sim = 0
            for i in range(len(fe_list)):
                fe=fe_list[i]
                sim = cosin_metric(fe, fe_2)
                if score_sim < sim:
                    id_sim = i
                    score_sim = sim
            if score_sim < THRESHOLD:
                id_sim = len(fe_list)
                fe_list.append(fe_2)
                score_sim = 0

            # display result
            fontScale = w / 512.0
            thickness = 2
            color = hsv_to_rgb(256 * id_sim / 16, 255, 255)
            cv2.rectangle(frame, top_left, bottom_right, color, 2)

            text_position = (int(fx)+4, int((fy+fh)-8))

            cv2.putText(
                frame,
                f"{id_sim} : {score_sim:5.3f}",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                color,
                thickness
            )

        cv2.imshow('frame', frame)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    if args.video:
        check_and_download_models(YOLOV3_FACE_WEIGHT_PATH, YOLOV3_FACE_MODEL_PATH, YOLOV3_FACE_REMOTE_PATH)

    if args.video is None:
        # still image mode
        # comparing two images specified args.inputs
        compare_images()
    else:
        # video mode
        # comparing the specified image and the video
        compare_image_and_video()


if __name__ == "__main__":
    main()
