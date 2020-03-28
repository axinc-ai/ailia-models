import sys
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../util')
from model_utils import check_and_download_models
from image_utils import load_image


# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = 'arcface.onnx'
MODEL_PATH = 'arcface.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/arcface/"

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
    '-i', '--inputs', metavar='IMAGEFILE_PATH',
    nargs=2,
    default=[IMG_PATH_1, IMG_PATH_2],
    help='Two image paths for calculating the face match.'
)
parser.add_argument(
    '-c', '--camera', metavar='IMAGEFILE_PATH',
    default=None,
    help='Compare the image loaded by web-camera with the specified ' +\
         'face image to determine if it is the same person or not.'
)
args = parser.parse_args()
            

# ======================
# Utils
# ======================
def preprocess_image(image):
    # (ref: https://github.com/ronghuaiyang/arcface-pytorch/issues/14)
    # use origin image and fliped image to infer,
    # and concat the feature as the final feature of the origin image.
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

    # compute execution time
    print('Start inference...')
    for i in range(5):
        start = int(round(time.time() * 1000))
        preds_ailia = net.predict(imgs)
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')

    # postprocessing
    fe_1 = np.concatenate([preds_ailia[0], preds_ailia[1]], axis=0)
    fe_2 = np.concatenate([preds_ailia[2], preds_ailia[3]], axis=0)
    sim = cosin_metric(fe_1, fe_2)
    
    print(f'Similarity of ({args.inputs[0]}, {args.inputs[1]}) : {sim:.3f}')
    if THRESHOLD > sim:
        print('They are not the same face!')
    else:
        print('They are the same face!')


def compare_image_and_webcamvideo():
    # prepare base image
    base_imgs = prepare_input_data(args.camera)

    # net itinialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # web camera
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("[Error] webcamera not found")
        sys.exit(1)

    _, frame = capture.read()
    # TODO need test when capture size is smaller than the rectangle
    frame_h, frame_w = frame.shape[0], frame.shape[1]
    rect_top = np.max((frame_h//2 - int(IMAGE_HEIGHT * 1.5), 0))
    rect_bottom = np.min((frame_h//2 + int(IMAGE_HEIGHT * 1.5), frame_h))
    rect_left = np.max((frame_w//2 - int(IMAGE_WIDTH * 1.5), 0))
    rect_right = np.min((frame_w//2 + int(IMAGE_WIDTH * 1.5), frame_w))
    
    # inference loop
    while(True):
        ret, original_frame = capture.read()                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        # preprocessing
        frame = original_frame[
            rect_top:rect_bottom + 1,
            rect_left:rect_right + 1,
        ]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        frame = preprocess_image(frame)
        input_data = np.concatenate([base_imgs, frame], axis=0)

        # inference
        preds_ailia = net.predict(input_data)

        # postprocessing
        fe_1 = np.concatenate([preds_ailia[0], preds_ailia[1]], axis=0)
        fe_2 = np.concatenate([preds_ailia[2], preds_ailia[3]], axis=0)
        sim = cosin_metric(fe_1, fe_2)
        bool_sim = False if THRESHOLD > sim else True
        
        cv2.putText(
            original_frame,
            f'Similarity: {sim:06.3f}  SAME PERSON: {bool_sim}',
            (50, 100),  # put text position
            cv2.FONT_HERSHEY_COMPLEX,  # font type
            1.2,  # font scale
            (255, 255, 255),  # font color
            thickness=2,
            lineType=cv2.LINE_AA
        )        
        # Draw a rectangle of the range to be inputted into the model
        cv2.rectangle(
            original_frame,
            (rect_left, rect_top),  # top left corner
            (rect_right, rect_bottom),  # bottom right corner
            (0, 255, 0),  # color
            3  # thickness
        )
        cv2.imshow('frame', original_frame)
        
    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')

    
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    
    if args.camera is None:
        # still image mode
        # comparing two images specified args.inputs
        compare_images()
    else:
        # video mode
        # comparing specified image and the image captured by web-camera
        compare_image_and_webcamvideo()


if __name__ == "__main__":
    main()
