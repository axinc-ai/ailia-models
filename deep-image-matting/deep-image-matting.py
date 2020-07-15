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
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deep-image-matting/'

SEGMENTATION_WEIGHT_PATH = 'u2net.onnx'
SEGMENTATION_MODEL_PATH = SEGMENTATION_WEIGHT_PATH + '.prototxt'
SEGMENTATION_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net/'

SEGMENTATION_WEIGHT_PATH = 'deeplabv3.opt.onnx'
SEGMENTATION_MODEL_PATH = SEGMENTATION_WEIGHT_PATH + '.prototxt'
SEGMENTATION_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deeplabv3/'

DEEPLABV3=True

#IMAGE_PATH = 'input.png'
#TRIMAP_PATH = 'trimap.png'

#IMAGE_PATH = 'couple.jpg'
#TRIMAP_PATH = ''

IMAGE_PATH = 'pixaboy.jpg'
TRIMAP_PATH = ''

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
img_rows = IMAGE_HEIGHT
img_cols = IMAGE_WIDTH

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

def postprocess(src_img, trimap, preds_ailia, use_trimap_directly=False):
    trimap = trimap[:,:,0].reshape((IMAGE_HEIGHT,IMAGE_WIDTH))
    preds_ailia = preds_ailia.reshape((IMAGE_HEIGHT,IMAGE_WIDTH))

    preds_ailia = preds_ailia * 255.0
    preds_ailia = get_final_output(preds_ailia,trimap)

    output_data = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,4))
    output_data[:,:,0]=src_img[:,:,0]
    output_data[:,:,1]=src_img[:,:,1]
    output_data[:,:,2]=src_img[:,:,2]
    if use_trimap_directly:
        output_data[:,:,3]=trimap
    else:
        output_data[:,:,3]=preds_ailia

    output_data[output_data>255]=255
    output_data[output_data<0]=0

    return output_data

# ======================
# Segmentation util
# ======================

import cv2
import numpy as np
from skimage import io
from PIL import Image


def transform(image, scaled_size):
    # RescaleT part in original repo
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = scaled_size*h/w, scaled_size
    else:
        new_h, new_w = scaled_size, scaled_size*w/h
    new_h, new_w = int(new_h), int(new_w)
    
    image = cv2.resize(image, (scaled_size, scaled_size))

    # ToTensorLab part in original repo
    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    image = image/np.max(image)
    if image.shape[2] == 1:
        tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
        tmpImg[:, :, 1] = (image[:, :, 0]-0.485)/0.229
        tmpImg[:, :, 2] = (image[:, :, 0]-0.485)/0.229
    else:
        tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
        tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
        tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225
    return tmpImg.transpose((2, 0, 1))[np.newaxis, :, :, :]


def load_image(image_path, scaled_size):
    image = io.imread(image_path)
    h, w = image.shape[0], image.shape[1]
    if 2 == len(image.shape):
        image = image[:, :, np.newaxis]
    return transform(image, scaled_size), h, w


def norm(pred):
    ma = np.max(pred)
    mi = np.min(pred)
    return (pred - mi) / (ma - mi)


def save_result(pred, savepath, srcimg_shape):
    """
    Parameters
    ----------
    srcimg_shape: (h, w)
    """
    # normalization
    pred = norm(pred)

    img = Image.fromarray(pred * 255).convert('RGB')
    img = img.resize(
        (srcimg_shape[1], srcimg_shape[0]),
        resample=Image.BILINEAR
    )
    img.save(savepath)
    

def gen_trimap(mask,k_size=(5,5),ite=1):
    kernel = np.ones(k_size,np.uint8)
    eroded = cv2.erode(mask,kernel,iterations = ite)
    dilated = cv2.dilate(mask,kernel,iterations = ite)
    trimap = np.full(mask.shape,128)
    trimap[eroded >= 254] = 255
    trimap[dilated <= 1] = 0
    return trimap


def generate_trimap(args):
    if DEEPLABV3:
        input_data = cv2.imread(args.input)
        w = input_data.shape[1]
        h = input_data.shape[0]
        input_data = cv2.resize(input_data, (513, 513))
        input_data = input_data / 127.5 - 1.0
        input_data = input_data.transpose((2,0,1))[np.newaxis, :, :, :]
    else:
        input_data, h, w = load_image(
            args.input,
            scaled_size=IMAGE_WIDTH,
        )

    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(SEGMENTATION_MODEL_PATH, SEGMENTATION_WEIGHT_PATH, env_id=env_id)

    preds_ailia = net.predict([input_data])

    if DEEPLABV3:
        pred = preds_ailia[0]
        pred = pred[0,15,:,:] / 21.0
    else:
        pred = preds_ailia[0][0, 0, :, :]

    save_result(pred, "debug_segmentation.png", [h, w])

    if DEEPLABV3:
        thre = 0.6
        pred [pred<thre] = 0
        pred [pred>=thre] = 1

    save_result(pred, "debug_segmentation_threshold.png", [h, w])

    if not DEEPLABV3:
        pred = norm(pred)

    img = Image.fromarray(pred * 255).convert('RGB')
    img = img.resize(
        (w, h),
        resample=Image.NEAREST#BILINEAR
    )

    trimap_data = np.asarray(img).copy()
    u2net_data = trimap_data.copy()

    trimap_data_original = trimap_data.copy()

    thre1=128
    thre2=128
    trimap_data[trimap_data_original<thre2] = 0
    trimap_data[trimap_data_original<thre1] = 0
    trimap_data[trimap_data_original>=thre2] = 255

    trimap_data = trimap_data.astype("uint8")

    trimap_data = gen_trimap(trimap_data,k_size=(7,7),ite=3)

    im = Image.fromarray(trimap_data.astype('uint8'))
    im.save("debug_trimap.png")

    return trimap_data,u2net_data


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    rgb_data = cv2.imread(args.input)
    src_img = cv2.imread(args.input)

    if args.trimap=="":
        trimap_data,u2net_data = generate_trimap(args)
    else:
        trimap_data = cv2.imread(args.trimap)
        u2net_data = trimap_data.copy()

    if IMAGE_PATH=="input.png":
        x = 320
        y = 0
        crop_size=(IMAGE_HEIGHT*2,IMAGE_WIDTH*2)
    else:
        x = 0
        y = 0
        crop_size=(IMAGE_HEIGHT,IMAGE_WIDTH)

    rgb_data=safe_crop(rgb_data, x, y, crop_size)
    trimap_data=safe_crop(trimap_data, x, y, crop_size)
    u2net_data=safe_crop(u2net_data, x, y, crop_size)
    src_img=safe_crop(src_img, x, y, crop_size)

    input_data = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,4))
    input_data[:,:,:,0:3] = rgb_data[:,:,0:3]

    input_data[:,:,:,3] = trimap_data[:,:,0]
    im = Image.fromarray(input_data.reshape((IMAGE_HEIGHT,IMAGE_WIDTH,4)).astype('uint8'))
    im.save("debug_input.png")
    input_data = input_data / 255.0
    
    # net initialize
    env_id = 0  # use cpu because overflow fp16 range
    #env_id = ailia.get_gpu_environment_id()   
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1,IMAGE_HEIGHT,IMAGE_WIDTH,4))

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

    # postprocessing
    res_img = postprocess(src_img, trimap_data, preds_ailia)
    cv2.imwrite(args.savepath, res_img)

    res_img = postprocess(src_img, u2net_data, preds_ailia, True)
    cv2.imwrite("debug_segmentation_alpha.png", res_img)

    res_img = postprocess(src_img, trimap_data, preds_ailia, True)
    cv2.imwrite("debug_trimap_alpha.png", res_img)

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
    if args.trimap == '':
        check_and_download_models(SEGMENTATION_WEIGHT_PATH, SEGMENTATION_MODEL_PATH, SEGMENTATION_REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
