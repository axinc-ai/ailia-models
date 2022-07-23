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
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

from face_restoration import get_face_landmarks_5

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'GFPGANv1.3.onnx'
MODEL_PATH = 'GFPGANv1.3.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/gfpgan/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 512

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'GFPGAN', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-d', '--detection',
    action='store_true',
    help='Use object detection.'
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

def preprocess(img):
    img = img[:, :, ::-1]  # BGR -> RGB
    img = normalize_image(img, normalize_type='127.5')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(pred):
    img = pred[0]
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img[:, :, ::-1]  # RGB -> BGR

    img = np.clip(img, -1, 1)
    img = (img + 1) * 127.5
    img = img.astype(np.uint8)

    return img


def predict(models, img):
    face_restor = models.get("face_restor")
    face_det = models.get("face_det")

    if not face_restor:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        cropped_faces = [img]
    else:
        all_landmarks_5 = get_face_landmarks_5(
            face_det, img, eye_dist_threshold=5)

        face_restor.clean_all()
        face_restor.read_image(img)
        # get face landmarks for each face
        face_restor.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
        # align and warp each face
        face_restor.align_warp_face()
        cropped_faces = face_restor.cropped_faces

    gfpgan = models["gfpgan"]

    # face restoration
    for cropped_face in cropped_faces:
        x = preprocess(cropped_face)

        # feedforward
        output = gfpgan.predict([x])
        pred = output[0]

        restored_face = post_processing(pred)

        if face_restor:
            face_restor.add_restored_face(restored_face)

    if face_restor:
        face_restor.get_inverse_affine(None)
        restored_img = face_restor.paste_faces_to_input_image(upsample_img=img)

    return restored_img


def recognize_from_image(models):
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
                restored_img = predict(models, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            restored_img = predict(models, img)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, restored_img)

    logger.info('Script finished successfully.')


def recognize_from_video(models):
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
        restored_img = predict(models, img)

        # show
        cv2.imshow('frame', restored_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(restored_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    WEIGHT_DET_PATH = "detection_Resnet50.onnx"
    MODEL_DET_PATH = "detection_Resnet50.onnx.prototxt"
    # DETECTION_MODEL_PATH = "blazeface.onnx.prototxt"
    # DETECTION_WEIGHT_PATH = "blazeface.onnx"
    # check_and_download_models(
    #     DETECTION_WEIGHT_PATH, DETECTION_MODEL_PATH,
    #     "https://storage.googleapis.com/ailia-models/blazeface/")

    env_id = args.env_id

    # initialize
    gfpgan = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    face_det = ailia.Net(MODEL_DET_PATH, WEIGHT_DET_PATH, env_id=env_id)

    models = {
        "gfpgan": gfpgan,
        "face_det": face_det,
    }

    upscale = 1

    if args.video or args.detection:
        import torch
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        face_restor = FaceRestoreHelper(
            upscale,
            face_size=IMAGE_SIZE,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=device)
        models["face_restor"] = face_restor

        # detector = ailia.Net(
        #     DETECTION_MODEL_PATH, DETECTION_WEIGHT_PATH, env_id=args.env_id
        # )
        # models["detector"] = detector

    if args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == '__main__':
    main()
