import os
import sys
import time
import argparse

import cv2
import numpy as np

import ailia
import onnxruntime

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = 'councilGAN-glasses.onnx'
MODEL_PATH = 'councilGAN-glasses.opt.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/councilGAN-glasses/"

IMAGE_PATH = 'sample.jpg'
SAVE_IMAGE_PATH = 'output.png'


# ======================
# Arguemnt Parser Config
# ======================

parser = argparse.ArgumentParser(
    description='Glasses removal GAN based on SimGAN'
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
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
parser.add_argument(
    '-o', '--onnx',
    action='store_true',
    help='Run on ONNXruntime instead of Ailia'
)
args = parser.parse_args()

# ======================
# Preprocessing functions
# ======================
def preprocess(image):
    image = center_crop(image)
    image = cv2.resize(image, (128, 128))
    # BGR to RGB
    image = image[...,::-1]
    # scale to [0,1]
    image = image/255.
    # swap channel order
    image = np.transpose(image, [2,0,1])
    # resize
    # normalize
    image = (image-0.5)/0.5
    return image.astype(np.float32)   

def center_crop(image):
    shape = image.shape[0:2]
    size = min(shape)
    return image[(shape[0]-size)//2:(shape[0]+size)//2, (shape[1]-size)//2:(shape[1]+size)//2, ...]

# ======================
# Postprocessing functions
# ======================
def postprocess_image(image):
    max_v = np.max(image)
    min_v = np.min(image)
    final_image = np.transpose((image-min_v)/(max_v-min_v)*255+0.5, (1,2,0)).round()
    out = np.clip(final_image, 0, 255).astype(np.uint8)
    return out

# ======================
# Main functions
# ======================
def process_image():
    # prepare input data
    img = preprocess(cv2.imread(args.input))
    
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

        # inference
        print('Start inference...')
        if args.benchmark:
            print('BENCHMARK mode')
            for i in range(5):
                data = [img[None,...]] 
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(data)[0][0]
                end = int(round(time.time() * 1000))
                print(f'\tailia processing time {end - start} ms')
        else:
            data = [img[None,...]] 
            preds_ailia = net.predict(data)
            preds_ailia = postprocess_image(preds_ailia[0][0])
    else:
        # teporary onnxruntime mode
        sess = onnxruntime.InferenceSession(WEIGHT_PATH)
        # inference
        print('Start inference in onnxruntime mode...')
        if args.benchmark:
            print('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))

                inputs = [i.name for i in sess.get_inputs()]
                outputs = [o.name for o in sess.get_outputs()]

                data = [img[None,...]] 
                out = sess.run(outputs, {i: data for i, data in zip(inputs, data)})

                preds_ailia = postprocess_image(out[0][0])
                end = int(round(time.time() * 1000))
                print(f'\tailia processing time {end - start} ms')
        else:
            inputs = [i.name for i in sess.get_inputs()]
            outputs = [o.name for o in sess.get_outputs()]

            data = [img[None,...]] 
            out = sess.run(outputs, {i: data for i, data in zip(inputs, data)})

            preds_ailia = postprocess_image(out[0][0])

    cv2.imwrite(args.savepath, preds_ailia[...,::-1])
    print('Script finished successfully.')
    

def process_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    if args.onnx:
        net = onnxruntime.InferenceSession('councilGAN-glasses.onnx')
    else:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

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
            
        img = preprocess(frame)
        
        if args.onnx:
            inputs = [i.name for i in net.get_inputs()]
            outputs = [o.name for o in net.get_outputs()]

            data = [img[None,...]] 
            out = net.run(outputs, {i: data for i, data in zip(inputs, data)})

            preds_ailia = postprocess_image(out[0][0])
        else:
            preds_ailia = postprocess_image(net.predict(img)[0][0])

        cv2.imshow('frame', preds_ailia[...,::-1])

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        process_video()
    else:
        # image mode
        process_image()


if __name__ == '__main__':
    main()
