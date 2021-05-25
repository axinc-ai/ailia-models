import sys, os
import time
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# import for dpt
import glob
import numpy as np
import util.io
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_MONODEPTH_PATH = "dpt_hybrid_monodepth.onnx"
MODEL_MONODEPTH_PATH = "dpt_hybrid_monodepth.onnx.prototxt"
WEIGHT_SEGMENTATION_PATH = "dpt_hybrid_segmentation.onnx"
MODEL_SEGMENTATION_PATH = "dpt_hybrid_segmentation.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/dense_prediction_transformers/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Dense Prediction Transformers',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '--task',
    required='monodepth',
    choices=['monodepth', 'segmentation'],
    help=('specify task you want to run.')
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img_raw):
    net_w = 576
    net_h = 384
    func = Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=False,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC)
    img = func({"image": img_raw})
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img = normalization(img)
    img = PrepareForNet()(img)
    img = img["image"]
    return img


def recognize_from_image(net):
    for i, img_name in enumerate(args.input):
        img_raw = util.io.read_image(img_name)
        img = preprocess(img_raw)

        sample = np.expand_dims(img,0)
        
        if args.onnx:
            input_name = net.get_inputs()[0].name
            prediction = net.run(None, {input_name: sample.astype(np.float32)})
            prediction = prediction[0]
        else:
            prediction = net.predict(sample)
            prediction = prediction[0]

        if args.task == 'monodepth':
            prediction = cv2.resize(prediction,(img_raw.shape[1],img_raw.shape[0]),interpolation=cv2.INTER_CUBIC)
        elif args.task == 'segmentation':
            scaled_predictin = np.zeros((prediction.shape[0],img_raw.shape[0],img_raw.shape[1]))
            for i in range(prediction.shape[0]):
                scaled_predictin[i] = cv2.resize(prediction[i],(img_raw.shape[1],img_raw.shape[0]),interpolation=cv2.INTER_CUBIC)
            prediction = np.argmax(scaled_predictin, axis=0) + 1

        savepath = get_savepath(args.savepath, img_name)
        logger.info(f'saved at : {savepath}')

        if args.task == 'monodepth':
            util.io.write_depth(savepath, prediction, bits=2)
        elif args.task == 'segmentation':
            util.io.write_segm_img(savepath, img_raw, prediction, alpha=0.5)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while(True):
        ret, img = capture.read()
        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (img_lab_orig, img_lab_rs) = preprocess(img)
        out = net.predict({'input.1': img_lab_rs})[0]
        out_img = post_process(out, img_lab_orig)
        out_img = np.array(out_img * 255, dtype=np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', out_img)

        # save results
        if writer is not None:
            writer.write(out_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')

def main():
    if args.task == 'monodepth':
        check_and_download_models(WEIGHT_MONODEPTH_PATH, MODEL_MONODEPTH_PATH, REMOTE_PATH)
        if args.onnx:
            import onnxruntime
            net = onnxruntime.InferenceSession(WEIGHT_MONODEPTH_PATH)
        else:
            net = ailia.Net(MODEL_MONODEPTH_PATH, WEIGHT_MONODEPTH_PATH, env_id=args.env_id)
    elif args.task == 'segmentation':
        if args.onnx:
            import onnxruntime
            net = onnxruntime.InferenceSession(WEIGHT_SEGMENTATION_PATH)
        else:
            net = ailia.Net(MODEL_SEGMENTATION_PATH, WEIGHT_SEGMENTATION_PATH, env_id=args.env_id)
        check_and_download_models(WEIGHT_SEGMENTATION_PATH, MODEL_SEGMENTATION_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)

if __name__ == '__main__':
    main()
