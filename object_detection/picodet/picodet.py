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
# from image_utils import load_image, normalize_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

from picodet_utils import grid_priors, get_bboxes, bbox2result

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'picodet_l_640_coco.onnx'
MODEL_PATH = 'picodet_l_640_coco.onnx.prototxt'
WEIGHT_XXX_PATH = 'xxx.onnx'
MODEL_XXX_PATH = 'xxx.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/picodet/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

THRESHOLD = 0.4
IOU = 0.45

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'XXX', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='object confidence threshold'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='IOU threshold for NMS'
)
parser.add_argument(
    '-m', '--model_type', default='xxx', choices=('xxx', 'XXX'),
    help='model type'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

# def draw_bbox(img, bboxes):
#     return img


# ======================
# Main functions
# ======================

def preprocess(img, image_shape):
    h, w = image_shape
    im_h, im_w, _ = img.shape

    # adaptive_resize
    scale = h / min(im_h, im_w)
    ow, oh = int(im_w * scale), int(im_h * scale)
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    img = np.array(Image.fromarray(img).resize((ow, oh), Image.BILINEAR))

    # center_crop
    if ow > w:
        x = (ow - w) // 2
        img = img[:, x:x + w, :]
    if oh > h:
        y = (oh - h) // 2
        img = img[y:y + h, :, :]

    img = normalize_image(img, normalize_type='ImageNet')

    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(output):
    cls_scores = output[:4]
    bbox_preds = output[4:]

    num_levels = len(cls_scores)

    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_priors = grid_priors(num_levels, featmap_sizes)

    cls_score_list = [
        cls_scores[i][0] for i in range(num_levels)
    ]
    bbox_pred_list = [
        bbox_preds[i][0] for i in range(num_levels)
    ]

    rescale = True
    det_bboxes, det_labels = get_bboxes(
        cls_score_list, bbox_pred_list, mlvl_priors,
        rescale, with_nms=True,
    )

    num_classes = 80
    bbox_results = bbox2result(det_bboxes, det_labels, num_classes)

    return bbox_results


def predict(net, img):
    shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    img = preprocess(img, shape)

    # feedforward
    output = net.predict([img])

    bbox_results = post_processing(output)

    return bbox_results


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(net, img)

        # res_img = draw_bbox(out)
        #
        # # plot result
        # savepath = get_savepath(args.savepath, image_path, ext='.png')
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = predict(net, img)

        # # plot result
        # res_img = draw_bbox(frame, out)
        #
        # # show
        # cv2.imshow('frame', res_img)
        # frame_shown = True
        #
        # # save results
        # if writer is not None:
        #     res_img = res_img.astype(np.uint8)
        #     writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # logger.info('=== Sample model ===')
    # dic_model = {
    #     'xxx': (WEIGHT_PATH, MODEL_PATH),
    #     'XXX': (WEIGHT_XXX_PATH, MODEL_XXX_PATH),
    # }
    # weight_path, model_path = dic_model[args.model_type]

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
