import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from image_utils import normalize_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

from egonet_utils import kpts2cs, cs2bbox
from egonet_utils import modify_bbox, get_affine_transform
from instance_utils import get_2d_3d_pair

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_HC_PATH = 'HC.onnx'
MODEL_HC_PATH = 'HC.onnx.prototxt'
WEIGHT_L_PATH = 'L.onnx'
MODEL_L_PATH = 'L.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/egonet/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 256

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'EgoNet', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def read_record(img, label_path, calib_path):
    list_2d, list_3d, list_id, pv, raw_bboxes = \
        get_2d_3d_pair(
            img,
            label_path, calib_path,
            augment=False, add_raw_bbox=True)

    bboxes = np.array([])
    if len(list_2d) != 0:
        for idx, kpts in enumerate(list_2d):
            list_2d[idx] = kpts.reshape(1, -1, 3)
            list_3d[idx] = list_3d[idx].reshape(1, -1, 3)
        all_keypoints_2d = np.concatenate(list_2d, axis=0)

        # compute 2D bounding box based on the projected 3D boxes
        bboxes_kpt = []
        for idx, keypoints in enumerate(all_keypoints_2d):
            # relatively tight bounding box: use enlarge = 1.0
            # delete invisible instances
            center, crop_size, _, _ = kpts2cs(
                keypoints[:, :2], enlarge=1.01)
            bbox = np.array(cs2bbox(center, crop_size))
            bboxes_kpt.append(np.array(bbox).reshape(1, 4))

        bboxes = np.vstack(bboxes_kpt)

    d = {'bbox_2d': bboxes}

    return d


# ======================
# Main functions
# ======================


def crop_single_instance(
        img,
        bbox,
        resolution):
    """
    Crop a single instance given an image and bounding box.
    """
    width, height = resolution
    target_ar = height / width
    ret = modify_bbox(bbox, target_ar)
    c, s, r = ret['c'], ret['s'], 0.

    # xy_dict: parameters for adding xy coordinate maps
    trans = get_affine_transform(c, s, r, (height, width))
    instance = cv2.warpAffine(
        img,
        trans,
        (int(resolution[0]), int(resolution[1])),
        flags=cv2.INTER_LINEAR
    )

    return instance


def crop_instances(img, annot_dict):
    resolution = [IMAGE_SIZE, IMAGE_SIZE]
    target_ar = resolution[1] / resolution[0]

    all_instances = []
    all_records = []
    boxes = annot_dict['bbox_2d']
    for idx, bbox in enumerate(boxes):
        # crop an instance with required aspect ratio
        instance = crop_single_instance(
            img, bbox, resolution)
        instance = normalize_image(instance, normalize_type='ImageNet')
        instance = instance.transpose(2, 0, 1)  # HWC -> CHW
        instance = np.expand_dims(instance, axis=0)
        instance = instance.astype(np.float32)
        all_instances.append(instance)

        ret = modify_bbox(bbox, target_ar)
        c, s, r = ret['c'], ret['s'], 0.
        all_records.append({
            'center': c,
            'scale': s,
            'bbox': bbox,
            'bbox_resize': ret['bbox'],
            'rotation': r,
        })

    all_instances = np.concatenate(all_instances, axis=0)

    return all_instances


def post_processing(output):
    return None


def predict(net_HC, net_L, img, annot_dict):
    instances = crop_instances(img, annot_dict)

    # feedforward
    if not args.onnx:
        output = net_HC.predict([instances])
    else:
        output = net_HC.run(None, {'input': instances})

    # pred = post_processing(output)
    pred = None

    return pred


def recognize_from_image(net_HC, net_L):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        label_path = "label_006272.txt"
        calib_path = "calib_006272.txt"
        annot_dict = read_record(img, label_path, calib_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                out = predict(net_HC, net_L, img, annot_dict)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out = predict(net_HC, net_L, img, annot_dict)

        # # plot result
        # savepath = get_savepath(args.savepath, image_path, ext='.png')
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('Checking HC model...')
    check_and_download_models(WEIGHT_HC_PATH, MODEL_HC_PATH, REMOTE_PATH)
    logger.info('Checking L model...')
    check_and_download_models(WEIGHT_L_PATH, MODEL_L_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net_HC = ailia.Net(MODEL_HC_PATH, WEIGHT_HC_PATH, env_id=env_id)
        net_L = ailia.Net(MODEL_L_PATH, WEIGHT_L_PATH, env_id=env_id)
    else:
        import onnxruntime
        net_HC = onnxruntime.InferenceSession(WEIGHT_HC_PATH)

    if args.video is not None:
        pass
    else:
        recognize_from_image(net_HC, net_L)


if __name__ == '__main__':
    main()
