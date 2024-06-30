import sys
import time

import ailia
import cv2
import numpy as np

import retinaface_utils as rut
from retinaface_utils import PriorBox

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

sys.path.append('../../generative_adversarial_networks/gfpgan')
from gfpgan import WEIGHT_PATH as GFPGAN_WEIGHT_PATH, MODEL_PATH as GFPGAN_MODEL_PATH, REMOTE_PATH as GFPGAN_REMOTE_PATH
from gfpgan import preprocess, post_processing, align_warp_face

logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
IMAGE_PATH = 'selfie.png'
SAVE_IMAGE_PATH = 'selfie_output.png'

MODEL_LISTS = ['resnet50', 'mobile0.25']
CONFIDENCE_THRES = 0.02
TOP_K = 5000
NMS_THRES = 0.4
KEEP_TOP_K = 750
VIS_THRES = 0.6

EYE_DIST_THRESH = 5

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'RetinaFace is a fast and powerful face detector.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='mobile0.25', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-r', '--rescale', metavar='RESCALE', type=float,
    default=1,
    help='scale down the original image size to prevent memory overflow, otherwise original size is used'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)

# ======================
# MODEL PARAMETERS
# ======================
if args.arch == 'resnet50':
    WEIGHT_PATH = 'retinaface_resnet50.onnx'
    MODEL_PATH = 'retinaface_resnet50.onnx.prototxt'
elif args.arch == 'mobile0.25':
    WEIGHT_PATH = 'retinaface_mobile0.25.onnx'
    MODEL_PATH = 'retinaface_mobile0.25.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/retinaface/"

def postprocessing(preds_ailia, input_data, cfg, dim):
    IMAGE_WIDTH, IMAGE_HEIGHT = dim
    scale = np.array([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT])
    loc, conf, landms = preds_ailia
    priorbox = PriorBox(cfg, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    priors = priorbox.forward()
    boxes = rut.decode(np.squeeze(loc, axis=0), priors, cfg['variance'])
    boxes = boxes * scale
    scores = np.squeeze(conf, axis=0)[:, 1]
    landms = rut.decode_landm(np.squeeze(landms, axis=0), priors, cfg['variance'])
    scale1 = np.array([input_data.shape[3], input_data.shape[2], input_data.shape[3], input_data.shape[2],
                            input_data.shape[3], input_data.shape[2], input_data.shape[3], input_data.shape[2],
                            input_data.shape[3], input_data.shape[2]])
    landms = landms * scale1

    # ignore low scores
    inds = np.where(scores > CONFIDENCE_THRES)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:TOP_K]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = rut.py_cpu_nms(dets, NMS_THRES)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:KEEP_TOP_K, :]
    landms = landms[:KEEP_TOP_K, :]

    detections = np.concatenate((dets, landms), axis=1)

    return detections




# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    if args.arch == "mobile0.25":
        cfg = rut.cfg_mnet
    elif args.arch == "resnet50":
        cfg = rut.cfg_re50
    mem_mode = ailia.get_memory_mode(reduce_constant=True, reuse_interstage=True)
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=mem_mode)
    resize = args.rescale

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        # org_img = load_image(image_path, (IMAGE_HEIGHT, IMAGE_WIDTH))
        org_img = imread(image_path, cv2.IMREAD_COLOR)

        # resize image
        IMAGE_WIDTH = int(org_img.shape[1] / resize)
        IMAGE_HEIGHT = int(org_img.shape[0] / resize)
        dim = (IMAGE_WIDTH, IMAGE_HEIGHT)
        org_img = cv2.resize(org_img, dim, interpolation = cv2.INTER_AREA)


        img = org_img - (104, 117, 123)
        input_data = img.transpose(2, 0, 1)
        input_data.shape = (1,) + input_data.shape

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict([input_data])
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            preds_ailia = net.predict([input_data])

        # post-processing
        detections = postprocessing(preds_ailia, input_data, cfg, dim)

        # generate detections
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        rut.plot_detections(org_img, detections, vis_thres=VIS_THRES, save_image_path=savepath)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            rut.save_json(json_file, detections, VIS_THRES)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    if args.arch == "mobile0.25":
        cfg = rut.cfg_mnet
    elif args.arch == "resnet50":
        cfg = rut.cfg_re50
    mem_mode = ailia.get_memory_mode(reduce_constant=True, reuse_interstage=True)
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=mem_mode)
    # disable FP16 for gfpgan
    if "FP16" in ailia.get_environment(args.env_id).props or sys.platform == 'Darwin':
        logger.warning('GFPGAN model do not work on FP16. So use CPU mode.')
        args.env_id = 0
    gfpgan = ailia.Net(GFPGAN_MODEL_PATH, GFPGAN_WEIGHT_PATH, env_id=args.env_id, memory_mode=mem_mode)
    resize = args.rescale

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # resize image
        IMAGE_WIDTH = int(frame.shape[1] / resize)
        IMAGE_HEIGHT = int(frame.shape[0] / resize)
        dim = (IMAGE_WIDTH, IMAGE_HEIGHT)
        input_image = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        img = input_image - (104, 117, 123)
        input_data = img.transpose(2, 0, 1)
        input_data.shape = (1,) + input_data.shape

        # inference
        preds_ailia = net.predict([input_data])

        # post-processing
        detections = postprocessing(preds_ailia, input_data, cfg, dim)
        cropped_faces = []
        bboxes = []
        all_landmarks_5 = []
        for det in detections:
            score = det[4]
            if score < VIS_THRES:
                continue
            det = list(map(int, det))

            # remove faces with too small eye distance: side faces or too small faces
            eye_dist = np.linalg.norm([det[6] - det[8], det[7] - det[9]])
            if eye_dist < EYE_DIST_THRESH:
                continue

            x1, y1, x2, y2 = det[:4]
            cropped_face = input_image[y1:y2, x1:x2]
            if cropped_face.size == 0:
                continue

            landmark = np.array([[det[i], det[i + 1]] for i in range(5, 15, 2)])

            # cropped_face = cv2.resize(cropped_face, (512, 512))
            # cropped_faces.append(cropped_face)
            cropped_face, _ = align_warp_face(input_image, [landmark])
            cropped_faces.append(cropped_face[0])

            bboxes.append((x1, y1, x2, y2, score))


        # run gfpgan
        restored_faces = []
        for cropped_face in cropped_faces:
            x = preprocess(cropped_face)

            # feedforward
            output = gfpgan.predict([x])
            pred = output[0]

            restored_face = post_processing(pred)
            restored_faces.append(restored_face)

        # display results
        # rut.plot_detections(input_image, detections, vis_thres=VIS_THRES)
        rut.plot_enhanced_faces(input_image, bboxes, cropped_faces, restored_faces)
        cv2.imshow('frame', input_image)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(input_image)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(GFPGAN_WEIGHT_PATH, GFPGAN_MODEL_PATH, GFPGAN_REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
