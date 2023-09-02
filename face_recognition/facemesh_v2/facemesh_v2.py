import sys
import time
import math
from logging import getLogger

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

import draw_utils

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'face_landmarks_detector.onnx'
MODEL_PATH = 'face_landmarks_detector.onnx.prototxt'
WEIGHT_XXX_PATH = 'xxx.onnx'
MODEL_XXX_PATH = 'xxx.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/facemesh_v2/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'FaceMesh-V2', IMAGE_PATH, SAVE_IMAGE_PATH
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

def draw_result(img, face_landmarks):
    # Draw the face landmarks.

    draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        connections=draw_utils.FACEMESH_TESSELATION,
        connection_drawing_spec=draw_utils.get_tesselation_style())

    draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        connections=draw_utils.FACEMESH_CONTOURS,
        connection_drawing_spec=draw_utils.get_contours_style())

    draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        connections=draw_utils.FACEMESH_IRISES,
        connection_drawing_spec=draw_utils.get_iris_connections_style())

    return img


def normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float,
        image_width: int, image_height: int):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) \
               and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x)
            and is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    return x_px, y_px


def draw_landmarks(
        image: np.ndarray,
        landmark_list,
        connections=None,
        connection_drawing_spec=None):
    # if not landmark_list:
    #     return

    image_rows, image_cols, _ = image.shape

    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list):
        x, y = landmark[:2]
        landmark_px = normalized_to_pixel_coordinates(
            x, y, image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    if connections:
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]

            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] \
                    if isinstance(connection_drawing_spec, dict) \
                    else connection_drawing_spec
                cv2.line(
                    image,
                    idx_to_coordinates[start_idx],
                    idx_to_coordinates[end_idx],
                    drawing_spec.color, drawing_spec.thickness)


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    # # adaptive_resize
    # scale = h / min(im_h, im_w)
    # ow, oh = int(im_w * scale), int(im_h * scale)
    # if ow != im_w or oh != im_h:
    #     img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
    #
    # img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BILINEAR))
    #
    # # center_crop
    # if ow > w:
    #     x = (ow - w) // 2
    #     img = img[:, x:x + w, :]
    # if oh > h:
    #     y = (oh - h) // 2
    #     img = img[y:y + h, :, :]

    img = cv2.imread("kekka_1.png")

    img = normalize_image(img, normalize_type='255')
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(input_tensors):
    num_landmarks = 478
    num_dimensions = 3

    input_tensors = input_tensors.reshape(-1)

    output_landmarks = np.zeros((num_landmarks, num_dimensions))
    for i in range(num_landmarks):
        offset = i * num_dimensions
        output_landmarks[i] = input_tensors[offset:offset + 3]

    norm_landmarks = output_landmarks / 256

    width = 0.429052
    height = 0.343577
    x_center = 0.489064
    y_center = 0.227319
    angle = 0.00830413

    for landmark in norm_landmarks:
        x = landmark[0] - 0.5
        y = landmark[1] - 0.5
        z = landmark[2]
        new_x = math.cos(angle) * x - math.sin(angle) * y
        new_y = math.sin(angle) * x + math.cos(angle) * y

        new_x = new_x * width + x_center
        new_y = new_y * height + y_center
        new_z = z * width

        landmark[...] = new_x, new_y, new_z

    return norm_landmarks


def predict(net, img):
    # shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'input_12': img})
    landmark_tensors, presence_flag_tensors, _ = output

    landmarks = post_processing(landmark_tensors)

    print(landmarks)
    print(landmarks.shape)

    return landmarks


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
                detection_result = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            detection_result = predict(net, img)

        res_img = draw_result(img, detection_result)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
