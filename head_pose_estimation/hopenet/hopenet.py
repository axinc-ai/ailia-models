import sys
import time

import cv2
import numpy as np

import ailia
import hopenet_utils as hut

sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


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
    'Hopenet, a head pose estimation.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-l', '--lite',
    action='store_true',
    help='With this option, a lite version of the head pose model is used.'
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
FACE_DETECTION_MODEL_NAME = 'blazeface'
HEAD_POSE_MODEL_NAME = 'hopenet_robust_alpha1'
if args.normal:
    FACE_DETECTION_WEIGHT_PATH = f'{FACE_DETECTION_MODEL_NAME}.onnx'
    FACE_DETECTION_MODEL_PATH = f'{FACE_DETECTION_MODEL_NAME}.onnx.prototxt'
    HEAD_POSE_WEIGHT_PATH = f'{HEAD_POSE_MODEL_NAME}.onnx'
    HEAD_POSE_MODEL_PATH = f'{HEAD_POSE_MODEL_NAME}.onnx.prototxt'
else:
    FACE_DETECTION_WEIGHT_PATH = f'{FACE_DETECTION_MODEL_NAME}.opt.onnx'
    FACE_DETECTION_MODEL_PATH = f'{FACE_DETECTION_MODEL_NAME}.opt.onnx.prototxt'
    HEAD_POSE_WEIGHT_PATH = f'{HEAD_POSE_MODEL_NAME}.opt.onnx'
    HEAD_POSE_MODEL_PATH = f'{HEAD_POSE_MODEL_NAME}.opt.onnx.prototxt'
FACE_DETECTION_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{FACE_DETECTION_MODEL_NAME}/'
HEAD_POSE_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/hopenet/'


# ======================
# Utils
# ======================
class HeadPoseEstimator:
    def __init__(self) -> None:
        # net initialize
        self.face_detector = ailia.Net(
            FACE_DETECTION_MODEL_PATH, FACE_DETECTION_WEIGHT_PATH, env_id=args.env_id
        )
        self.hp_estimator = ailia.Net(
            HEAD_POSE_MODEL_PATH, HEAD_POSE_WEIGHT_PATH, env_id=args.env_id
        )

    def predict(self, img):
        """
        Estimate the head pose given an image.

        Parameters
        ----------
        img: NumPy array
            The image to resize (BGR channels).

        Returns
        -------
        head_pose: NumPy array
            Head pose(s) in radians. Roll (left+), yaw (right+), pitch (down+)
            values are given in the detected person's frame of reference.
        """
        # Face detection
        input_face_det, scale, padding = hut.face_detector_preprocess(img)
        preds_det = self.face_detector.predict([input_face_det])
        detections = hut.face_detector_postprocess(preds_det)

        input_hp_est, centers, theta = hut.head_pose_preprocess(img, detections, scale, padding)
        self.hp_estimator.set_input_shape(input_hp_est.shape)
        preds_hp = self.hp_estimator.predict([input_hp_est])
        head_poses = hut.head_pose_postprocess(preds_hp, theta)
        return head_poses, centers

    def _get_rot_mat(self, axis, angle):
        rot_mat = np.zeros((3, 3), dtype=np.float32)
        if axis == 'z':
            i = 2
        elif axis == 'y':
            i = 1
        elif axis == 'x':
            i = 0
        else:
            raise ValueError(f'Axis {axis} is not a valid argument.')

        rot_mat[i, i] = 1
        rot_mat[i-1, i-1] = np.cos(angle)
        rot_mat[i-1, i-2] = np.sin(angle)
        rot_mat[i-2, i-1] = -np.sin(angle)
        rot_mat[i-2, i-2] = np.cos(angle)
        return rot_mat

    def draw(self, img, head_poses, centers, horizontal_flip=False):
        hp, c = head_poses[0], centers[0]
        new_img = img[:, ::-1].copy()
        rot_mat = self._get_rot_mat('z', hp[0])
        rot_mat = rot_mat @ self._get_rot_mat('y', hp[1])
        rot_mat = rot_mat @ self._get_rot_mat('x', hp[2])
        hp_vecs = rot_mat.T # Each row is rotated x, y, z respectively
        
        if horizontal_flip:
            hp_vecs[0, 1] *= -1
            hp_vecs[1:, 0] *= -1
            c[0] = new_img.shape[1] - c[0]

        for i, vec in enumerate(hp_vecs):
            tip = tuple((c + 100 * vec[:2]).astype(int))
            color = [0, 0, 0]
            color[i] = 255
            cv2.arrowedLine(new_img, tuple(c.astype(int)), tip, tuple(color), thickness=2)
        # cv2.imshow('', new_img)
        # cv2.waitKey(0)
        return new_img
    
    def predict_and_draw(self, img):
        head_poses, centers = self.predict(img)
        return self.draw(img, head_poses, centers)


# ======================
# Main functions
# ======================
def recognize_from_image():
    hp_estimator = HeadPoseEstimator()

    # input image loop
    for image_path in args.input:
        logger.info(image_path)
        src_img = cv2.imread(image_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for _ in range(5):
                start = int(round(time.time() * 1000))
                img_draw = hp_estimator.predict_and_draw(src_img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            img_draw = hp_estimator.predict_and_draw(src_img)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img_draw)
    logger.info('Script finished successfully.')


def recognize_from_video():
    hp_estimator = HeadPoseEstimator()

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w, fps=capture.get(cv2.CAP_PROP_FPS))
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        preds = hp_estimator.predict(frame)
        frame_draw = hp_estimator.draw(frame, *preds)

        if args.video == '0': # Flip horizontally if camera
            # visual_img = np.ascontiguousarray(visual_img[:,::-1])
            visual_img = hp_estimator.draw(frame, *preds, horizontal_flip=True)
        else:
            visual_img = frame_draw

        cv2.imshow('frame', visual_img)

        # save results
        if writer is not None:
            writer.write(frame_draw)

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(
        FACE_DETECTION_WEIGHT_PATH, FACE_DETECTION_MODEL_PATH, FACE_DETECTION_REMOTE_PATH
    )
    check_and_download_models(
        HEAD_POSE_WEIGHT_PATH, HEAD_POSE_MODEL_PATH, HEAD_POSE_REMOTE_PATH
    )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
