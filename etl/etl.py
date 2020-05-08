import sys
import time
import argparse

import cv2

import codecs

import ailia
# import original modules
sys.path.append('../util')
from utils import check_file_existance
from model_utils import check_and_download_models
from image_utils import load_image
from webcamera_utils import adjust_frame_size


# ======================
# PARAMETERS
# ======================
MODEL_PATH = 'etl_BINARY_squeezenet128_20.prototxt'
WEIGHT_PATH = 'etl_BINARY_squeezenet128_20.caffemodel'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/etl/'

IMAGE_PATH = 'font.png'
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
ETL_PATH = 'etl_BINARY_squeezenet128_20.txt'
MAX_CLASS_COUNT = 3
SLEEP_TIME = 3  # for webcam mode


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Japanese character classification model.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGEFILE_PATH',
    default=IMAGE_PATH, 
    help='The input image path.'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='The input video path. ' +\
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
args = parser.parse_args()


# ======================
# Utils
# ======================
def preprocess_image(img):
    if img.shape[2] == 3 :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1 : 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    img = cv2.bitwise_not(img)
    return img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    etl_word = codecs.open(ETL_PATH, 'r', 'utf-8').readlines()
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    img = preprocess_image(img)
    
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32
    )

    # compute execution time
    for i in range(5):
        start = int(round(time.time() * 1000))
        classifier.compute(img, MAX_CLASS_COUNT)
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')

    # get result
    count = classifier.get_class_count()
    print(f'class_count: {count}')

    for idx in range(count) :
        print(f"+ idx={idx}")
        info = classifier.get_class(idx)
        print(f"  category={info.category} [ {etl_word[info.category]} ]" )
        print(f"  prob={info.prob}")


def recognize_from_video():
    etl_word = codecs.open(ETL_PATH, 'r', 'utf-8').readlines()
    
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32
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

        in_frame, frame = adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)
        frame = preprocess_image(frame)
        
        # inference
        # compute execution time
        classifier.compute(frame, MAX_CLASS_COUNT)
        
        # get result
        count = classifier.get_class_count()

        print('==============================================================')
        print(f'class_count: {count}')
        for idx in range(count) :
            print(f"+ idx={idx}")
            info = classifier.get_class(idx)
            print(f"  category={info.category} [ {etl_word[info.category]} ]" )
            print(f"  prob={info.prob}")
        cv2.imshow('frame', in_frame)
        time.sleep(SLEEP_TIME)
            
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
