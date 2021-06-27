import sys
import time

import cv2
import numpy as np

import ailia

sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from animalpose_utils import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

MODEL_LIST = []
WEIGHT_HRNET_W32_PATH = 'hrnet_w32_256x256.onnx'
MODEL_HRNET_W32_PATH = 'hrnet_w32_256x256.onnx.prototxt'
WEIGHT_HRNET_W48_PATH = 'hrnet_w48_256x256.onnx'
MODEL_HRNET_W48_PATH = 'hrnet_w48_256x256.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/animalpose/'

# IMAGE_PATH = 'ho105.jpeg'
IMAGE_PATH = 'ca110.jpeg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 256

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    '2D animal_pose estimation',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model', metavar='ARCH',
    default='heavy', choices=MODEL_LIST,
    help='Set model architecture: ' + ' | '.join(MODEL_LIST)
)
parser.add_argument(
    '-th', '--threshold',
    default=0.5, type=float,
    help='The detection threshold'
)
args = update_parser(parser)


# ======================
# Utils
# ======================

def preprocess(img):
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255
    img = np.expand_dims(img, axis=0)

    return img


def postprocess(output, img_metas):
    """Decode keypoints from heatmaps.

    Args:
        output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        img_metas (list(dict)): Information about data augmentation
            By default this includes:
            - "image_file: path to the image file
            - "center": center of the bbox
            - "scale": scale of the bbox
            - "rotation": rotation of the bbox
            - "bbox_score": score of bbox
    """
    batch_size = len(img_metas)

    if 'bbox_id' in img_metas[0]:
        bbox_ids = []
    else:
        bbox_ids = None

    c = np.zeros((batch_size, 2), dtype=np.float32)
    s = np.zeros((batch_size, 2), dtype=np.float32)
    score = np.ones(batch_size)
    for i in range(batch_size):
        c[i, :] = img_metas[i]['center']
        s[i, :] = img_metas[i]['scale']

        if 'bbox_score' in img_metas[i]:
            score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
        if bbox_ids is not None:
            bbox_ids.append(img_metas[i]['bbox_id'])

    preds, maxvals = keypoints_from_heatmaps(output, c, s)

    all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
    all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals
    all_boxes[:, 0:2] = c[:, 0:2]
    all_boxes[:, 2:4] = s[:, 0:2]
    all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
    all_boxes[:, 5] = score

    result = {}

    result['preds'] = all_preds
    result['boxes'] = all_boxes
    result['bbox_ids'] = bbox_ids

    return result


def vis_pose_result(img, result):
    palette = np.array([
        [255, 128, 0], [255, 153, 51], [255, 178, 102],
        [230, 230, 0], [255, 153, 255], [153, 204, 255],
        [255, 102, 255], [255, 51, 255], [102, 178, 255],
        [51, 153, 255], [255, 153, 153], [255, 102, 102],
        [255, 51, 51], [153, 255, 153], [102, 255, 102],
        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
        [255, 255, 255]
    ])
    skeleton = [[1, 2], [1, 3], [2, 4], [1, 5], [2, 5], [5, 6], [6, 8],
                [7, 8], [6, 9], [9, 13], [13, 17], [6, 10], [10, 14],
                [14, 18], [7, 11], [11, 15], [15, 19], [7, 12], [12, 16],
                [16, 20]]
    pose_limb_color = palette[[0] * 20]
    pose_kpt_color = palette[[0] * 20]

    img = show_result(
        img,
        result,
        skeleton,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color)

    return img


# ======================
# Main functions
# ======================

def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = src_img = load_image(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = np.load("ca.npy")
        img = np.expand_dims(img, axis=0)

        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                # Pose estimation
                start = int(round(time.time() * 1000))
                output = net.predict([img])
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            # inference
            output = net.predict([img])

        heatmap = output[0]

        img_metas = [{
            'center': np.array([155., 132.], dtype=np.float32),
            'scale': np.array([1.775, 1.775], dtype=np.float32),
        }]
        result = postprocess(heatmap, img_metas)
        pose = result['preds'][0]

        # pose_results = []
        # for pose, person_result, bbox_xyxy in zip(
        #         poses, person_results, bboxes_xyxy
        # ):
        #     pose_result = person_result.copy()
        #     pose_result['keypoints'] = pose
        #     pose_result['bbox'] = bbox_xyxy
        #     pose_results.append(pose_result)

        # plot result
        # src_img = cv2.cvtColor(src_img, cv2.COLOR_BGRA2BGR)
        pose_results = [{
            'bbox': np.array([13., 36., 296., 227.]),
            'keypoints': pose,
        }, ]
        img = vis_pose_result(src_img, pose_results)

        # save results
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img)

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

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # inference
        pass

        cv2.imshow('frame', frame)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    # logger.info('=== animalpose model ===')
    # info = {
    #     'lite': (WEIGHT_LITE_PATH, MODEL_LITE_PATH),
    #     'full': (WEIGHT_FULL_PATH, MODEL_FULL_PATH),
    #     'heavy': (WEIGHT_HEAVY_PATH, MODEL_HEAVY_PATH),
    # }
    # weight_path, model_path = info[args.model]
    # check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # load model
    env_id = ailia.get_gpu_environment_id()
    logger.info(f'env_id: {env_id}')

    model_path = "hrnet_w32_256x256.onnx.prototxt"
    weight_path = "hrnet_w32_256x256.onnx"
    net = ailia.Net(model_path, weight_path, env_id=env_id)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
