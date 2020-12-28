import sys
import time
import argparse

import cv2
import numpy as np

import ailia
import facemesh_utils as fut

sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'man.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Face Mesh, an on-device real-time face recognition.', IMAGE_PATH, SAVE_IMAGE_PATH,
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
DETECTION_MODEL_NAME = 'blazeface'
LANDMARK_MODEL_NAME = 'facemesh'
# if args.normal:
DETECTION_WEIGHT_PATH = f'{DETECTION_MODEL_NAME}.onnx'
DETECTION_MODEL_PATH = f'{DETECTION_MODEL_NAME}.onnx.prototxt'
LANDMARK_WEIGHT_PATH = f'{LANDMARK_MODEL_NAME}.onnx'
LANDMARK_MODEL_PATH = f'{LANDMARK_MODEL_NAME}.onnx.prototxt'
# else:
    # DETECTION_WEIGHT_PATH = f'{DETECTION_MODEL_NAME}.opt.onnx'
    # DETECTION_MODEL_PATH = f'{DETECTION_MODEL_NAME}.opt.onnx.prototxt'
    # LANDMARK_WEIGHT_PATH = f'{LANDMARK_MODEL_NAME}.opt.onnx'
    # LANDMARK_MODEL_PATH = f'{LANDMARK_MODEL_NAME}.opt.onnx.prototxt'
DETECTION_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{DETECTION_MODEL_NAME}/'
LANDMARK_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{LANDMARK_MODEL_NAME}/'


# ======================
# Utils
# ======================
def draw_roi(img, roi):
    for i in range(roi.shape[0]):
        (x1,x2,x3,x4), (y1,y2,y3,y4) = roi[i]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0,255,0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0,0,0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0,0,0), 2)


def draw_landmarks(img, points, color=(0, 0, 255), size=2):
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, thickness=cv2.FILLED)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    src_img = cv2.imread(args.input)
    _, img128, scale, pad = fut.resize_pad(src_img[:,:,::-1])
    input_data = img128.astype('float32') / 127.5 - 1.0
    input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Net(DETECTION_MODEL_PATH, DETECTION_WEIGHT_PATH, env_id=env_id)
    estimator = ailia.Net(LANDMARK_MODEL_PATH, LANDMARK_WEIGHT_PATH, env_id=env_id)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for _ in range(5):
            start = int(round(time.time() * 1000))
            # # Palm detection
            # preds = detector.predict([input_data])
            # detections = fut.detector_postprocess(preds)

            # # Hand landmark estimation
            # presence = [0, 0] # [left, right]
            # if detections[0].size != 0:
            #     imgs, affines, _ = fut.estimator_preprocess(src_img, detections, scale, pad)
            #     estimator.set_input_shape(imgs.shape)
            #     detections, confidences = estimator.predict([imgs])

            #     # postprocessing
            #     # detections[0:-1:3] *= self.x_scale
            #     # detections[1:-1:3] *= self.y_scale
            #     landmarks = fut.denormalize_landmarks(detections, affines)
            #     for i in range(len(flags)):
            #         landmark, flag, handed = landmarks[i], flags[i], 1 - handedness[i]
            #         if flag > 0.75:
            #             if handed < 0.5: # Right handedness when not flipped camera input
            #                 presence[0] = 1
            #             else:
            #                 presence[1] = 1
            #             draw_landmarks(src_img, landmark[:,:2], fut.HAND_CONNECTIONS, size=2)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        # Face detection
        preds = detector.predict([input_data])
        detections = fut.detector_postprocess(preds)

        # Face landmark estimation
        if detections[0].size != 0:
            imgs, affines, box = fut.estimator_preprocess(src_img[:,:,::-1], detections, scale, pad)
            draw_roi(src_img, box)
            estimator.set_input_shape(imgs.shape)
            landmarks, confidences = estimator.predict([imgs])
            normalized_landmarks = landmarks / 192.0

            # postprocessing
            landmarks = fut.denormalize_landmarks(normalized_landmarks, affines)
            for i in range(len(landmarks)):
                landmark, confidence = landmarks[i], confidences[i]
                # if confidence > 0: # Can be > 1, no idea what it represents
                draw_landmarks(src_img, landmark[:,:2], size=1)

    cv2.imwrite(args.savepath, src_img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Net(DETECTION_MODEL_PATH, DETECTION_WEIGHT_PATH, env_id=env_id)
    estimator = ailia.Net(LANDMARK_MODEL_PATH, LANDMARK_WEIGHT_PATH, env_id=env_id)

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        frame = np.ascontiguousarray(frame[:,::-1,:])
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        img256, _, scale, pad = fut.resize_pad(frame[:,:,::-1])
        input_data = img256.astype('float32') / 255.
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        # Palm detection
        preds = detector.predict([input_data])
        detections = fut.detector_postprocess(preds)

        # Hand landmark estimation
        presence = [0, 0] # [left, right]
        if detections[0].size != 0:
            img, affine, _ = fut.estimator_preprocess(frame, detections, scale, pad)
            estimator.set_input_shape(img.shape)
            flags, handedness, normalized_landmarks = estimator.predict([img])

            # postprocessing
            landmarks = fut.denormalize_landmarks(normalized_landmarks, affine)
            for i in range(len(flags)):
                landmark, flag, handed = landmarks[i], flags[i], handedness[i]
                if flag > 0.75:
                    if handed > 0.5:
                        presence[0] = 1
                    else:
                        presence[1] = 1
                    draw_landmarks(frame, landmark[:,:2], fut.HAND_CONNECTIONS, size=2)

        if presence[0] and presence[1]:
            text = 'Left and right'
        elif presence[0]:
            text = 'Left'
        elif presence[1]:
            text = 'Right'
        else:
            text = 'No hand'
        cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow('frame', frame)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')
    pass


def main():
    # model files check and download
    check_and_download_models(DETECTION_WEIGHT_PATH, DETECTION_MODEL_PATH, DETECTION_REMOTE_PATH)
    check_and_download_models(LANDMARK_WEIGHT_PATH, LANDMARK_MODEL_PATH, LANDMARK_REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
