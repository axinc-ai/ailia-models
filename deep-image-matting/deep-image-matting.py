import sys
import time
import argparse

import numpy as np
import cv2

import ailia
# import original modules
sys.path.append('../util')
from webcamera_utils import preprocess_frame  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import check_file_existance  # noqa: E402


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'deep-image-matting.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'
REMOTE_PATH =\
    'https://storage.googleapis.com/ailia-models/deep-image-matting/'

IMAGE_PATH = 'input.png'
TRIMAP_PATH = 'trimap.png'

SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Deep Image Matting'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=IMAGE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-t', '--trimap', metavar='IMAGE',
    default=TRIMAP_PATH,
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
args = parser.parse_args()


# ======================
# Utils
# ======================
img_rows = 320
img_cols = 320

def safe_crop(mat, x, y, crop_size=(img_rows, img_cols)):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != (img_rows, img_cols):
        ret = cv2.resize(ret, dsize=(img_rows, img_cols), interpolation=cv2.INTER_NEAREST)
    return ret

def get_final_output(out, trimap):
    unknown_code = 128
    mask = np.equal(trimap, unknown_code).astype(np.float32)
    return (1 - mask) * trimap + mask * out

def postprocess(src_img, trimap, preds_ailia):
    print(trimap.shape)
    print(preds_ailia.shape)
    #trimap=trimap.transpose(0,2,3,1)
    
    trimap = trimap[:,:,0].reshape((320,320))
    preds_ailia = preds_ailia.reshape((320,320))

    #trimap=trimap.transpose(0,2,3,1)
    preds_ailia = preds_ailia * 255.0
    preds_ailia = get_final_output(preds_ailia,trimap)
    print(preds_ailia.shape)

    print(src_img.shape)
    print(preds_ailia)

    output_data = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,4))
    output_data[:,:,0]=src_img[:,:,0]
    output_data[:,:,1]=src_img[:,:,1]
    output_data[:,:,2]=src_img[:,:,2]
    output_data[:,:,3]=preds_ailia

    output_data[output_data>255]=255
    output_data[output_data<0]=0

    return output_data
    #preds_ailia.reshape()
    #return preds_ailia * 255

    pred = sigmoid(preds_ailia)[0][0]
    mask = pred >= 0.5

    mask_n = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    mask_n[:, :, 0] = 255
    mask_n[:, :, 0] *= mask

    image_n = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)

    # discard padded area
    h, w, _ = image_n.shape
    delta_h = h - IMAGE_HEIGHT
    delta_w = w - IMAGE_WIDTH

    top = delta_h // 2
    bottom = IMAGE_HEIGHT - (delta_h - top)
    left = delta_w // 2
    right = IMAGE_WIDTH - (delta_w - left)

    mask_n = mask_n[top:bottom, left:right, :]
    image_n = image_n * 0.5 + mask_n * 0.5
    return image_n


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    rgb_data = cv2.imread(args.input)
    trimap_data = cv2.imread(args.trimap)
    src_img = cv2.imread(args.input)

    x = 320
    y = 0
    crop_size=(IMAGE_HEIGHT*2,IMAGE_WIDTH*2)

    rgb_data=safe_crop(rgb_data, x, y, crop_size)
    trimap_data=safe_crop(trimap_data, x, y, crop_size)
    src_img=safe_crop(src_img, x, y, crop_size)

    print(rgb_data.shape)
    print(trimap_data.shape)
    #trimap_data = trimap_data[:,0,:,:].reshape((1,320,320,1))
    input_data = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,4))
    print(input_data.shape)
    input_data[:,:,:,0:3] = rgb_data[:,:,0:3]# / 255.0
    input_data[:,:,:,3] = trimap_data[:,:,0]# / 255.0
    #print(input_data)
    #image_n = cv2.cvtColor(input_data, cv2.COLOR_RGBA2RGBA)
    #input_data = input_data.transpose(0,2,3,1)
    from PIL import Image
    im = Image.fromarray(input_data.reshape((320,320,4)).astype('uint8'))
    im.save("input_dump.png")
    input_data = input_data / 255.0
    print(input_data.shape)
    #print(input_data)
    #print(trimap_data[0,1,160,160])

    # net initialize
    env_id = 0#ailia.get_gpu_environment_id()
    # overflow fp16 range
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1,320,320,4))

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')    
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(input_data)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = net.predict(input_data)

    print(preds_ailia.shape)
    
    # postprocessing
    res_img = postprocess(src_img, trimap_data, preds_ailia)
    cv2.imwrite(args.savepath, res_img)
    print('Script finished successfully.')


def recognize_from_video():
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

        src_img, input_data = preprocess_frame(
            frame,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            normalize_type='255'
        )

        src_img = cv2.resize(src_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

        preds_ailia = net.predict(input_data)

        res_img = postprocess(src_img, preds_ailia)
        cv2.imshow('frame', res_img / 255.0)

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
