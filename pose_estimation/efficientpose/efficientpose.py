import sys
import time
import argparse
from os.path import join, normpath

import cv2
from PIL import Image, ImageDraw
import numpy as np

import ailia
import efficientpose_utils as e_utils

sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

import onnxruntime

#TODO:More refactoring on threshold

# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'MPII.jpg'
SAVE_IMAGE_PATH = 'output.png'

MODEL_VARIANTS = ['rt','i','ii','iii','iv']
MODEL_VARIANT = 'rt'

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'EfficientPose,.', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-i', '--input', type=str,
    default=IMAGE_PATH,
    help='The input image for pose estimation.'
)
parser.add_argument(
    '-v', '--video', type=str,
    help='The input video for pose estimation.'
)
parser.add_argument(
    '-m', '--model_variant', type=str,
    default=MODEL_VARIANT, choices=MODEL_VARIANTS,
    help="The model variant for pose estimation, 'rt','i','ii','iii','iv'."
)
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
args = update_parser(parser)

RESOLUTION = {'rt': 224, 'i': 256, 'ii': 368, 'iii': 480, 'iv': 600}[args.model_variant]

# ======================
# Parameters 2
# ======================
MODEL_NAME = 'EfficientPose{}'.format(args.model_variant.upper())

WEIGHT_PATH = f'{MODEL_NAME}.onnx'
MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'

REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/efficientpose/'

# ======================
# Utils
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def line(input_img, coordinates, point1, point2):
    threshold = 0.3
    if coordinates[point1][3] > threshold and\
       coordinates[point2][3] > threshold:
        color = hsv_to_rgb(255*point1/ailia.POSE_KEYPOINT_CNT, 255, 255)

        x1 = int(input_img.shape[1] * coordinates[point1][1])
        y1 = int(input_img.shape[0] * coordinates[point1][2])
        x2 = int(input_img.shape[1] * coordinates[point2][1])
        y2 = int(input_img.shape[0] * coordinates[point2][2])
        cv2.line(input_img, (x1, y1), (x2, y2), color, 5)

def display_result(input_img, coordinates):
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_HEAD_TOP,e_utils.EFFICIENT_POSE_KEYPOINT_UPPER_NECK)
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_UPPER_NECK,e_utils.EFFICIENT_POSE_KEYPOINT_THORAX)
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_THORAX,e_utils.EFFICIENT_POSE_KEYPOINT_RIGHT_SHOULDER)
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_THORAX,e_utils.EFFICIENT_POSE_KEYPOINT_LEFT_SHOULDER)
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_THORAX,e_utils.EFFICIENT_POSE_KEYPOINT_PELVIS)

    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_RIGHT_SHOULDER,e_utils.EFFICIENT_POSE_KEYPOINT_RIGHT_ELBOW)
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_RIGHT_ELBOW,e_utils.EFFICIENT_POSE_KEYPOINT_RIGHT_WRIST)
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_LEFT_SHOULDER,e_utils.EFFICIENT_POSE_KEYPOINT_LEFT_ELBOW)
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_LEFT_ELBOW,e_utils.EFFICIENT_POSE_KEYPOINT_LEFT_RIGHT_WRIST)

    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_PELVIS,e_utils.EFFICIENT_POSE_KEYPOINT_RIGHT_HIP)
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_PELVIS,e_utils.EFFICIENT_POSE_KEYPOINT_LEFT_HIP)
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_RIGHT_HIP,e_utils.EFFICIENT_POSE_KEYPOINT_RIGHT_KNEE)
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_RIGHT_KNEE,e_utils.EFFICIENT_POSE_KEYPOINT_RIGHT_ANKLE)
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_LEFT_HIP,e_utils.EFFICIENT_POSE_KEYPOINT_LEFT_KNEE)
    line(input_img,coordinates,e_utils.EFFICIENT_POSE_KEYPOINT_LEFT_KNEE,e_utils.EFFICIENT_POSE_KEYPOINT_LEFT_ANKLE)

def annotate_image(file_path, coordinates):
    """
    Annotates supplied image from predicted coordinates.
    
    Args:
        file_path: path
            System path of image to annotate
        coordinates: list
            Predicted body part coordinates for image
    """
    
    # Load raw image
    image = Image.open(file_path)
    image_width, image_height = image.size
    image_side = image_width if image_width >= image_height else image_height

    # Annotate image
    image_draw = ImageDraw.Draw(image)
    image_coordinates = coordinates[0]
    image = e_utils.display_body_parts(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, marker_radius=int(image_side/150))
    image = e_utils.display_segments(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, segment_width=int(image_side/100))
    
    # Save annotated image
    image.save(normpath(file_path.split('.')[0] + '_tracked.png'))

# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    src_img = cv2.imread(args.input)
    image_height, image_width = src_img.shape[:2]
    batch = np.expand_dims(src_img[...,::-1], axis=0)

    # Preprocess batch
    batch = e_utils.preprocess(batch, RESOLUTION)

    # net initialize
    if args.onnx:
        model = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        env_id = ailia.get_gpu_environment_id()
        print(f'env_id: {env_id}')
        model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
 
    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for _ in range(5):
            start = int(round(time.time() * 1000))
            _ = model.predict([batch])[0]
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        # Person detection
        print('batch.shape', batch.shape)
        if args.onnx:
            ort_inputs = {model.get_inputs()[0].name: batch.astype(np.float32)}
            model_out = model.run(None, ort_inputs)[0]
        else:
            model_out = model.predict([batch])[0]
        print('model_out.shape',model_out.shape)
        # Extract coordinates
        coordinates = [e_utils.extract_coordinates(model_out[0,...], image_height, image_width)]

    display_result(src_img, coordinates[0])
    # image.save(normpath(file_path.split('.')[0] + '_out.png'))
    cv2.imwrite(normpath(args.input.split('.')[0] + '_out.png'), src_img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    if args.onnx:
        model = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        env_id = ailia.get_gpu_environment_id()
        print(f'env_id: {env_id}')
        model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    capture = get_capture(args.video)

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # prepare input data
        image_height, image_width = frame.shape[:2]
        batch = [frame[...,::-1]]
        # batch = np.expand_dims(frame, axis=0)

        # Preprocess batch
        batch = e_utils.preprocess(batch, RESOLUTION)

        # inference
        # Person detection
        if args.onnx:
            ort_inputs = {model.get_inputs()[0].name: batch.astype(np.float32)}
            model_out = model.run(None, ort_inputs)[0]
            # print('ONNX model_out.shape',model_out.shape)
        else:
            model_out = model.predict([batch])[0]
            # print('ailia model_out.shape',model_out.shape)
        # Extract coordinates
        coordinates = e_utils.extract_coordinates(model_out[0,...], image_height, image_width, real_time=True)

        display_result(frame, coordinates)
        cv2.imshow('frame', frame)
        # e_utils.display_camera(cv2, frame, coordinates, image_height, image_width)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')
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
