import sys
import time
import json
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import preprocess_frame  # noqa: E402C


# ======================
# Parameters
# ======================
TAG_WEIGHT_PATH = 'illust2vec_tag_ver200.onnx'
TAG_MODEL_PATH = 'illust2vec_tag_ver200.onnx.prototxt'
FE_WEIGHT_PATH = 'illust2vec_ver200.onnx'
FE_MODEL_PATH = 'illust2vec_ver200.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ill2vec/'
TAG_PATH = 'tag_list.json'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='illustration2vec model'
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
args = parser.parse_args()


# ======================
# Utils
# ======================
def prepare_input_data(img):
    mean = np.array([164.76139251,  167.47864617,  181.13838569])

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float32)
    img -= mean
    img = img.transpose((2, 0, 1))
    return img[np.newaxis, :, :, :]


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    input_img = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None',
    )
    input_data = prepare_input_data(input_img)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    tag_net = ailia.Net(TAG_MODEL_PATH, TAG_WEIGHT_PATH, env_id=env_id)
    tag_net.set_input_shape(input_data.shape)

    if check_file_existance(TAG_PATH):
        tags = json.loads(open(TAG_PATH, 'r').read())
        assert(len(tags) == 1539)

    input_dict = {'data': input_data}

    # DEBUG ============
    import onnxruntime
    onnx_net = onnxruntime.InferenceSession(TAG_WEIGHT_PATH, None)
    y = onnx_net.run(['prob'], input_dict)[0]
    print(f'[DEBUG] onnxruntime output shape: {y.shape}')
    # DEBUG ============
            
    # Running the inference on the same input five times
    # to measure execution performance
    for i in range(5):
        start = int(round(time.time() * 1000))
        preds_ailia = tag_net.predict(input_dict)

        # blob version
        # input_blobs = tag_net.get_input_blob_list()
        # tag_net.set_input_blob_data(input_data, input_blobs[0])
        # tag_net.update()
        # preds_ailia = tag_net.get_results()[0]
        
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')

    # postprocessing
    print(f'[DEBUG] preds_ailia.shape: {preds_ailia.shape}')
    print('Script finished successfully.')


def recognize_from_video():
    # Pass for now
    """
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
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

        input_image, input_data = preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='127.5'
        )
        # ???

        # inference
        # 1.
        preds_ailia = net.predict(input_data)

        # 2.
        input_blobs = net.get_input_blob_list()
        net.set_input_blob_data(input_data, input_blobs[0])
        net.update()
        preds_ailia = net.get_results()

        # postprocessing
        # ???
        cv2.imshow('frame', input_image)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')
    """


def main():
    # model files check and download
    check_and_download_models(TAG_WEIGHT_PATH, TAG_MODEL_PATH, REMOTE_PATH)
    # check_and_download_models(FE_WEIGHT_PATH, FE_MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
