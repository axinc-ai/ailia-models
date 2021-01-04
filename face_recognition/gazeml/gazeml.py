import sys
import time
import platform

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
sys.path.append('../../face_detection/blazeface')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402
from blazeface_utils import compute_blazeface_with_keypoint  # noqa: E402

# ======================
# PARAMETERS
# ======================
# MODEL_PATH = "gazeml_elg_i180x108_n64.onnx.prototxt"
# WEIGHT_PATH = "gazeml_elg_i180x108_n64.onnx"
# OUTPUT_BLOB_NAME = "import/hourglass/hg_3/after/hmap/conv/BiasAdd:0"

WEIGHT_PATH = 'gazeml_elg_i60x36_n32.onnx'
MODEL_PATH = 'gazeml_elg_i60x36_n32.onnx.prototxt'
OUTPUT_BLOB_NAME = "import/hourglass/hg_2/after/hmap/conv/BiasAdd:0"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/gazeml/"

IMAGE_PATH = 'eye.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 36  # 108
IMAGE_WIDTH = 60  # 180

THRESHOLD = 0.1
SCALE = 1.2

FACE_WEIGHT_PATH = 'blazeface.onnx'
FACE_MODEL_PATH = 'blazeface.onnx.prototxt'
FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
FACE_MARGIN = 1.2


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('gaze estimation model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)


# ======================
# Utils
# ======================
def plot_on_image(img, preds_ailia, eye_x, eye_y, eye_w, eye_h):
    for i in range(preds_ailia.shape[3]):
        probMap = preds_ailia[0, :, :, i]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (eye_w * point[0]) / preds_ailia.shape[2] * SCALE
        y = (eye_h * point[1]) / preds_ailia.shape[1] * SCALE
        color = (0, 255, 255)
        if i >= 8:
            color = (255, 0, 0)
        if i >= 16:
            color = (0, 0, 255)
        if prob > THRESHOLD:
            cv2.circle(
                img,
                (int(x + eye_x), int(y + eye_y)),
                3,
                color,
                thickness=-1,
                lineType=cv2.FILLED
            )
        # print(f'[DEBUG]  x: {int(x):3d}\ty: {int(y):3d}\tprob:{prob}')


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    org_img = cv2.imread(args.input)
    img = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        rgb=False,
        normalize_type='None'
    )
    img = cv2.equalizeHist(img)
    if platform.system() == 'Darwin':  # For Mac OS (FP16)
        data = img[np.newaxis, np.newaxis, :, :] / 255.0 - 0.5
    else:
        data = img[np.newaxis, np.newaxis, :, :] / 127.5 - 1.0
    eyeI = np.concatenate((data, data), axis=0)
    eyeI = eyeI.reshape(2, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(eyeI)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = net.predict(eyeI)

    preds_ailia = net.get_blob_data(
        net.find_blob_index_by_name(OUTPUT_BLOB_NAME)
    )

    # postprocessing
    plot_on_image(
        org_img, preds_ailia, 0, 0, org_img.shape[1], org_img.shape[0]
    )
    cv2.imwrite(args.savepath, org_img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # detect eyes
        detections, keypoints = compute_blazeface_with_keypoint(
            detector,
            frame,
            anchor_path='../../face_detection/blazeface/anchors.npy',
        )

        eye_list = []
        for keypoint in keypoints:
            lx = int(keypoint["eye_left_x"] * frame.shape[1])
            ly = int(keypoint["eye_left_y"] * frame.shape[0])
            rx = int(keypoint["eye_right_x"] * frame.shape[1])
            ry = int(keypoint["eye_right_y"] * frame.shape[0])

            eye_w = abs((lx-rx)/2)
            eye_h = eye_w * IMAGE_HEIGHT / IMAGE_WIDTH

            eye_list.append([int(lx-eye_w/2), int(ly-eye_h/2),
                             int(lx+eye_w/2), int(ly+eye_h/2)])
            eye_list.append([int(rx-eye_w/2), int(ry-eye_h/2),
                             int(rx+eye_w/2), int(ry+eye_h/2)])

        # detect eye keypoints
        for eye_position in eye_list:
            color = (255, 255, 255)
            top_left = (eye_position[0], eye_position[1])
            bottom_right = (eye_position[2], eye_position[3])
            cv2.rectangle(frame, top_left, bottom_right, color, thickness=2)

            # prepare frame
            crop_img = frame[
                eye_position[1]:eye_position[3],
                eye_position[0]:eye_position[2],
            ]
            img, resized_img = webcamera_utils.adjust_frame_size(
                crop_img, IMAGE_HEIGHT, IMAGE_WIDTH
            )
            data = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            data = cv2.equalizeHist(data)
            if platform.system() == 'Darwin':
                data = data[np.newaxis, np.newaxis, :, :] / 255.0 - 0.5
            else:
                data = data[np.newaxis, np.newaxis, :, :] / 127.5 - 1.0
            eyeI = np.concatenate((data, data), axis=0)
            eyeI = eyeI.reshape(2, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

            # inference
            preds_ailia = net.predict(eyeI)
            preds_ailia = net.get_blob_data(
                net.find_blob_index_by_name(OUTPUT_BLOB_NAME)
            )

            # postprocessing
            plot_on_image(
                frame,
                preds_ailia,
                eye_position[0],
                eye_position[1],
                eye_position[2]-eye_position[0],
                eye_position[3]-eye_position[1],
            )

        cv2.imshow('frame', frame)
        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video:
        check_and_download_models(
            FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH
        )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
