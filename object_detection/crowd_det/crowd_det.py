import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image, plot_results  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
from nms_utils import nms_boxes
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'rcnn_fpn_baseline_mge.onnx'
MODEL_PATH = 'rcnn_fpn_baseline_mge.onnx.prototxt'
WEIGHT_XXX_PATH = 'xxx.onnx'
MODEL_XXX_PATH = 'xxx.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/crowd_det/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Detection in Crowded Scenes', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-m', '--model_type', default='xxx', choices=('xxx', 'XXX'),
    help='model type'
)
parser.add_argument(
    '-t', '--threshold', type=float, default=0.3,
    help='threshold'
)
parser.add_argument(
    '-it', '--iou_threshold', type=float, default=0.5,
    help='iou_threshold'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    short_size, max_size = 800, 1400

    im_h, im_w, _ = img.shape

    im_min = min(im_h, im_w)
    im_max = max(im_h, im_w)
    scale = short_size / im_min
    if scale * im_max > max_size:
        scale = max_size / im_max
    ow, oh = int(round(im_w * scale)), int(round(im_h * scale))
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, scale


def post_processing(pred_boxes):
    pred_cls_threshold = args.threshold
    nms_thres = args.iou_threshold

    pred_boxes = pred_boxes.reshape(-1, 6)
    keep = pred_boxes[:, 4] > pred_cls_threshold
    pred_boxes = pred_boxes[keep]
    keep = nms_boxes(pred_boxes, pred_boxes[:, 4], nms_thres)
    pred_boxes = pred_boxes[keep]

    return pred_boxes


def convert_to_detector_object(preds, im_w, im_h):
    detector_object = []
    for i in range(len(preds)):
        (x1, y1, w, h) = preds[i, :4]
        score = float(preds[i, 4])

        r = ailia.DetectorObject(
            category="",
            prob=score,
            x=x1 / im_w,
            y=y1 / im_h,
            w=w / im_w,
            h=h / im_h,
        )
        detector_object.append(r)

    return detector_object


def predict(net, img):
    im_h, im_w, _ = img.shape
    img, scale = preprocess(img)
    _, _, oh, ow = img.shape
    im_info = np.array([[oh, ow, scale, im_h, im_w, 0]])

    # img = np.load("image1.npy")
    # im_info = np.load("im_info1.npy")
    # print(img)
    # print(img.shape)
    # print(im_info)
    # print(im_info.shape)

    # feedforward
    if not args.onnx:
        output = net.predict([img, im_info])
    else:
        output = net.run(None, {'img': img, 'im_info': im_info})

    pred_boxes = output[0]

    pred_boxes = post_processing(pred_boxes)
    pred_boxes[:, :4] /= scale
    pred_boxes[:, 2:4] -= pred_boxes[:, :2]
    print(pred_boxes)
    print(pred_boxes.shape)

    return pred_boxes


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
                pred = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net, img)

        # plot result
        detect_object = convert_to_detector_object(pred, img.shape[1], img.shape[0])
        res_img = plot_results(detect_object, img)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

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
        pred = predict(net, img)

        # plot result
        detect_object = convert_to_detector_object(pred, img.shape[1], img.shape[0])
        res_img = plot_results(detect_object, img)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            res_img = res_img.astype(np.uint8)
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'xxx': (WEIGHT_PATH, MODEL_PATH),
        'XXX': (WEIGHT_XXX_PATH, MODEL_XXX_PATH),
    }
    # weight_path, model_path = dic_model[args.model_type]
    weight_path, model_path = dic_model['xxx']

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
