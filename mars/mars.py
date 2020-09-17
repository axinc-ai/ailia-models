import os
import sys
import time
import argparse
import re
from collections import deque
 
import numpy as np
import cv2
import onnxruntime

import ailia

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size  # noqa: E402C
from image_utils import load_image, normalize_image, draw_result_on_img  # noqa: E402C

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'mars.onnx'
MODEL_PATH = 'mars.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mars/'

IMAGE_PATH = 'inputs'
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
DURATION = 16

HMDB51_LABEL = [
'brush_hair',
'cartwheel',
'catch',
'chew',
'clap',
'climb',
'climb_stairs',
'dive',
'draw_sword',
'dribble',
'drink',
'eat',
'fall_floor',
'fencing',
'flic_flac',
'golf',
'handstand',
'hit',
'hug',
'jump',
'kick',
'kick_ball',
'kiss',
'laugh',
'pick',
'pour',
'pullup',
'punch',
'push',
'pushup',
'ride_bike',
'ride_horse',
'run',
'shake_hands',
'shoot_ball',
'shoot_bow',
'shoot_gun',
'sit',
'situp',
'smile',
'smoke',
'somersault',
'stand',
'swing_baseball',
'sword',
'sword_exercise',
'talk',
'throw',
'turn',
'walk',
'wave',
]


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='MARS model'
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
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
parser.add_argument(
    '-d', '--duration', metavar='DURATION',
    default=DURATION,
    help='Sampling duration.',
    type=int
)
parser.add_argument(
    '-t', '--top', metavar='TOP',
    default=3,
    help='Number of outputs for category.',
    type=int
)
parser.add_argument(
    '-a', '--ailia',
    action='store_true',
    help='Use ailia SDK'
)
args = parser.parse_args()

# ======================
# Main functions
# ======================
def print_mars_result(result):
    indexes = result[0,:].argsort()[::-1]
    print('==============================================================')
    print(f'class_count={args.top}')
    for i, idx in enumerate(indexes[0:args.top]):
        print(f'+ idx={i}')
        print(f'  category={idx}['
            f'{HMDB51_LABEL[idx]} ]')
        print(f'  prob={result[0, idx]}')

def recognize_from_image():
    # prepare input data
    num = lambda val : int(re.sub("\\D", "", val))
    sorted_inputs_path = sorted(os.listdir(args.input), key=num)
    input_blob = np.empty((1, 3, args.duration, IMAGE_HEIGHT, IMAGE_WIDTH))
    for i, input_path in enumerate(sorted_inputs_path[0:args.duration]):
        img = load_image(
            args.input + '/' + input_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='None',
            gen_input_ailia=True
        )
        input_blob[0, :, i, :, :] = img
    next_input_index = args.duration
    input_frame_size = len(sorted_inputs_path)

    # # net initialize
    if args.ailia:
        env_id = ailia.get_gpu_environment_id()
        print(f'env_id: {env_id}')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        net.set_input_shape((1, 3, args.duration, IMAGE_HEIGHT, IMAGE_WIDTH))
    else:
        # # net initialize(onnxruntime)
        session = onnxruntime.InferenceSession(WEIGHT_PATH)
        input_name = session.get_inputs()[0].name 
        output_name = session.get_outputs()[0].name

    # inferece
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            if args.ailia:
                result = net.predict(input_blob)
            else:
                result = session.run([output_name], {input_name: input_blob.astype(np.float32)})[0]
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        while(next_input_index < input_frame_size):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if args.ailia:
                result = net.predict(input_blob)
            else:
                result = session.run([output_name], {input_name: input_blob.astype(np.float32)})[0]

            print_mars_result(result)
            
            preview_img = cv2.imread(args.input + '/' + sorted_inputs_path[next_input_index - args.duration])
            cv2.imshow('preview', preview_img)

            for i in range(args.duration - 1):
                input_blob[0, :, i, :, :] = input_blob[0, :, i + 1, :, :]
            
            img = load_image(
                args.input + '/' + sorted_inputs_path[next_input_index],
                (IMAGE_HEIGHT, IMAGE_WIDTH),
                normalize_type='None',
                gen_input_ailia=True
            )
            input_blob[0, :, args.duration - 1, :, :] = img
            next_input_index += 1
        
    print('Script finished successfully.')


def convert_input_frame(frame):
    frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    frame = frame[np.newaxis, :, :, :]
    return frame


def recognize_from_video():
    if args.video == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    else:
        if check_file_existance(args.video):
            capture = cv2.VideoCapture(args.video)

    # # net initialize
    if args.ailia:
        env_id = ailia.get_gpu_environment_id()
        print(f'env_id: {env_id}')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        net.set_input_shape((1, 3, args.duration, IMAGE_HEIGHT, IMAGE_WIDTH))
    else:
        # net initialize(onnxruntime)
        session = onnxruntime.InferenceSession(WEIGHT_PATH)
        input_name = session.get_inputs()[0].name 
        output_name = session.get_outputs()[0].name

    # prepare input data
    original_queue = deque([])
    input_blob = np.empty((1, 3, args.duration, IMAGE_HEIGHT, IMAGE_WIDTH))
    for i in range(args.duration - 1):
        ret, frame = capture.read()
        original_queue.append(frame)
        input_blob[0, :, i, :, :] = convert_input_frame(frame)
    
    next_input_index = args.duration - 1
    input_frame_size = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    while(next_input_index <= input_frame_size):        
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue
        original_queue.append(frame)
        input_blob[0, :, args.duration - 1, :, :] = convert_input_frame(frame)
        
        if args.ailia:
            result = net.predict(input_blob)
        else:
            result = session.run([output_name], {input_name: input_blob.astype(np.float32)})[0]
        print_mars_result(result)
        
        preview_img = original_queue.popleft()
        cv2.imshow('preview', preview_img)

        for i in range(args.duration - 1):
            input_blob[0, :, i, :, :] = input_blob[0, :, i + 1, :, :]
        
        next_input_index+=1 

    capture.release()
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
