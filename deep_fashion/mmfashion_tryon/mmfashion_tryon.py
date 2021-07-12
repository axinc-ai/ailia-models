import sys, os
import time
import json

import numpy as np
import cv2
from PIL import Image, ImageDraw

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from mmfashion_tryon_utils import *
from pose_utils import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_GMM_PATH = './GMM_epoch_40.onnx'
MODEL_GMM_PATH = './GMM_epoch_40.onnx.prototxt'
WEIGHT_TOM_PATH = './TOM_epoch_40.onnx'
MODEL_TOM_PATH = './TOM_epoch_40.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mmfashion_vto/'

WEIGHT_YOLOV3_PATH = 'yolov3.opt2.onnx'
MODEL_YOLOV3_PATH = 'yolov3.opt2.onnx.prototxt'
REMOTE_YOLOV3_PATH = 'https://storage.googleapis.com/ailia-models/yolov3/'

WEIGHT_POSE_PATH = 'pose_resnet_50_256x192.onnx'
MODEL_POSE_PATH = 'pose_resnet_50_256x192.onnx.prototxt'
REMOTE_POSE_PATH = 'https://storage.googleapis.com/ailia-models/pose_resnet/'

IMAGE_CLOTH_PATH = 'cloth/019029_1.jpg'
IMAGE_PERSON_PATH = 'image/000320_0.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 192

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('MMFashion Virtual Try-on model', IMAGE_CLOTH_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-p', '--person', metavar='IMAGE', default=IMAGE_PERSON_PATH,
    help='Image of person.'
)
parser.add_argument(
    '-pp', '--parse', metavar='IMAGE/DIR', default='image-parse/',
    help='Parsed image of person image. If a directory name is specified, '
         'Search for a png file with the same name as the person image file'
)
parser.add_argument(
    '-k', '--keypoints', metavar='FILE/DIR', default=None,
    help='Keypoints json file. If a directory name is specified, '
         'find a json file with the same name as the person image file'
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

def preprocess(img):
    mean = np.array((0.5,) * img.shape[2])
    std = np.array((0.5,) * img.shape[2])

    img = img / 255
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW

    return img.astype(np.float32)


def post_processing(data):
    data = (data + 1) * 0.5 * 255
    data = np.clip(data, 0, 255)
    data = data.transpose(1, 2, 0)  # CHW -> HWC

    img = data.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def pose_estimation(det_net, pose_net, img):
    h, w = img.shape[:2]

    THRESHOLD = 0.4
    IOU = 0.45
    det_net.compute(img, THRESHOLD, IOU)

    CATEGORY_PERSON = 0
    count = det_net.get_object_count()
    a = sorted([
        det_net.get_object(i) for i in range(count)
    ], key=lambda x: x.prob, reverse=True)
    bbox = [
        (w * obj.x, h * obj.y, w * obj.w, h * obj.h)
        for obj in a if obj.category == CATEGORY_PERSON
    ][0]

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[0] + bbox[2]
    y1 = bbox[0] + bbox[3]
    x0, y0, x1, y1 = keep_aspect(
        x0, y0, x1, y1, h, w
    )
    img = img[y0:y1, x0:x1, :]

    offset_x = x0 / w
    offset_y = y0 / h
    scale_x = img.shape[1] / w
    scale_y = img.shape[0] / h

    POSE_HEIGTH = 256
    POSE_WIDTH = 192
    img = cv2.resize(img, (POSE_WIDTH, POSE_HEIGTH))

    # BGR format
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img / 255.0 - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    output = pose_net.predict(img)

    center = np.array([POSE_WIDTH / 2, POSE_HEIGTH / 2], dtype=np.float32)
    scale = np.array([1, 1], dtype=np.float32)
    preds, maxvals = get_final_preds(output, [center], [scale])

    ailia_to_mpi = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1, -1
    ]
    pose = []
    for j in range(ailia.POSE_KEYPOINT_CNT):
        i = ailia_to_mpi[j]
        if j == ailia.POSE_KEYPOINT_BODY_CENTER:
            x = (preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_LEFT], 0] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_RIGHT], 0] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_HIP_LEFT], 0] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_HIP_RIGHT], 0]) / 4
            y = (preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_LEFT], 1] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_RIGHT], 1] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_HIP_LEFT], 1] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_HIP_RIGHT], 1]) / 4
        elif j == ailia.POSE_KEYPOINT_SHOULDER_CENTER:
            x = (preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_LEFT], 0] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_RIGHT], 0]) / 2
            y = (preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_LEFT], 1] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_RIGHT], 1]) / 2
        else:
            x = preds[0, i, 0]
            y = preds[0, i, 1]

        pose.append([
            x / POSE_WIDTH * scale_x + offset_x,
            y / POSE_HEIGTH * scale_y + offset_y,
            0
        ])

    pose = np.array(pose)

    return pose[:18]


# ======================
# Main functions
# ======================

def cloth_agnostic(det_net, pose_net, seg_net, img):
    fine_height = IMAGE_HEIGHT
    fine_width = IMAGE_WIDTH
    radius = 5

    h, w = img.shape[:2]

    # fine_height = h
    # fine_width = w

    if det_net:
        pose_data = pose_estimation(det_net, pose_net, img)
        pose_data = pose_data * [fine_width, fine_height, 1]
    else:
        person_path = args.person
        name = os.path.splitext(os.path.basename(person_path))[0]
        parse_path = (os.path.join(args.parse, '%s.png' % name) \
                          if os.path.isdir(args.parse) else args.parse) \
            if args.parse else '%s_parse.png' % name
        pose_path = (os.path.join(args.keypoints, '%s_keypoints.json' % name) \
                         if os.path.isdir(args.keypoints) else args.keypoints) \
            if args.keypoints else '%s_keypoints.json' % name

        # load pose points
        with open(pose_path, 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

    # person image
    im = preprocess(img)

    # parsing image
    im_parse = np.array(Image.open(parse_path))
    parse_shape = (im_parse > 0).astype(np.float32)
    phead = sum([
        (im_parse == i).astype(np.float32) for i in [1, 2, 4, 13]
    ])

    # shape downsample
    parse_shape = (parse_shape * 255).astype(np.uint8)
    parse_shape = Image.fromarray(parse_shape)
    parse_shape = parse_shape.resize(
        (fine_width // 16, fine_height // 16), Image.BILINEAR)
    parse_shape = parse_shape.resize(
        (fine_width, fine_height), Image.BILINEAR)
    parse_shape = np.expand_dims(np.asarray(parse_shape), axis=2)
    shape = preprocess(parse_shape)

    # upper cloth
    im_h = im * phead - (1 - phead)  # [-1,1], fill 0 for other parts

    point_num = pose_data.shape[0]
    pose_map = np.zeros((point_num, fine_height, fine_width), dtype=np.float32)
    r = radius
    im_pose = Image.new('L', (fine_width, fine_height))
    pose_draw = ImageDraw.Draw(im_pose)
    for i in range(point_num):
        one_map = Image.new('L', (fine_width, fine_height))
        draw = ImageDraw.Draw(one_map)
        pointx = pose_data[i, 0]
        pointy = pose_data[i, 1]
        if pointx > 1 and pointy > 1:
            draw.rectangle(
                (pointx - r, pointy - r, pointx + r, pointy + r), 'white',
                'white')
            pose_draw.rectangle(
                (pointx - r, pointy - r, pointx + r, pointy + r), 'white',
                'white')
        one_map = np.expand_dims(np.asarray(one_map), axis=2)
        one_map = preprocess(one_map)
        pose_map[i] = one_map[0]

    agnostic = np.vstack([shape, im_h, pose_map])

    return agnostic


def predict(GMM_net, TOM_net, cloth, agnostic):
    if not args.onnx:
        output = GMM_net.predict({
            'cloth': cloth, 'agnostic': agnostic
        })
    else:
        in0 = GMM_net.get_inputs()[0].name
        in1 = GMM_net.get_inputs()[1].name
        out0 = GMM_net.get_outputs()[0].name
        output = GMM_net.run(
            [out0],
            {in0: cloth, in1: agnostic})

    grid = output[0]
    warped_cloth = grid_sample(cloth, grid, padding_mode='border')

    if not args.onnx:
        output = TOM_net.predict({
            'cloth': warped_cloth, 'agnostic': agnostic
        })
    else:
        in0 = TOM_net.get_inputs()[0].name
        in1 = TOM_net.get_inputs()[1].name
        out0 = TOM_net.get_outputs()[0].name
        output = TOM_net.run(
            [out0],
            {in0: warped_cloth, in1: agnostic})

    tryon = output[0]

    return tryon, warped_cloth


def recognize_from_image(GMM_net, TOM_net, det_net, pose_net, seg_net):
    img = load_image(args.person)

    agnostic = cloth_agnostic(det_net, pose_net, seg_net, img)
    agnostic = np.expand_dims(agnostic, axis=0)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare cloth image
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = preprocess(img)
        img = np.expand_dims(img, axis=0)

        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(GMM_net, TOM_net, img, agnostic)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            output = predict(GMM_net, TOM_net, img, agnostic)

        tryon, warped_cloth = output
        tryon = post_processing(tryon[0])
        warped_cloth = post_processing(warped_cloth[0])

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, tryon)

        savepath_warp = '%s-warp-cloth%s' % os.path.splitext(savepath)
        logger.info(f'saved at : {savepath_warp}')
        cv2.imwrite(savepath_warp, warped_cloth)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('=== GMM model ===')
    check_and_download_models(WEIGHT_GMM_PATH, MODEL_GMM_PATH, REMOTE_PATH)
    logger.info('=== TOM model ===')
    check_and_download_models(WEIGHT_TOM_PATH, MODEL_TOM_PATH, REMOTE_PATH)
    if not args.keypoints:
        logger.info('=== detector model ===')
        check_and_download_models(WEIGHT_YOLOV3_PATH, MODEL_YOLOV3_PATH, REMOTE_YOLOV3_PATH)
        logger.info('=== pose model ===')
        check_and_download_models(WEIGHT_POSE_PATH, MODEL_POSE_PATH, REMOTE_POSE_PATH)

    # initialize
    if args.onnx:
        import onnxruntime
        GMM_net = onnxruntime.InferenceSession(WEIGHT_GMM_PATH)
        TOM_net = onnxruntime.InferenceSession(WEIGHT_TOM_PATH)
    else:
        GMM_net = ailia.Net(MODEL_GMM_PATH, WEIGHT_GMM_PATH, env_id=args.env_id)
        TOM_net = ailia.Net(MODEL_TOM_PATH, WEIGHT_TOM_PATH, env_id=args.env_id)

    # initialize
    if not args.keypoints:
        det_net = ailia.Detector(
            MODEL_YOLOV3_PATH,
            WEIGHT_YOLOV3_PATH,
            80,
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
            env_id=args.env_id,
        )
        pose_net = ailia.Net(
            MODEL_POSE_PATH, WEIGHT_POSE_PATH, env_id=args.env_id)
    else:
        det_net = pose_net = None

    seg_net = None
    recognize_from_image(GMM_net, TOM_net, det_net, pose_net, seg_net)


if __name__ == '__main__':
    main()
