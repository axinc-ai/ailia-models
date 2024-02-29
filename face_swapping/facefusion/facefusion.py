import onnx
import ailia
import shutil
import numpy as np
from onnx import numpy_helper

from facefusion_utils.face_analyser import get_one_face, get_average_face
from facefusion_utils.utils import read_static_images, read_static_image
from facefusion_utils.face_store import append_reference_face
from facefusion_utils.face_swapper import pre_process, get_reference_frame, post_process
from facefusion_utils.face_swapper import process_image as _process_image
from facefusion_utils.ffmpeg import compress_image


# ======================
# Parameters
# ======================

WEIGHT_FACE_DETECTOR_PATH = 'yoloface_8n.onnx'
MODEL_FACE_DETECTOR_PATH = 'yoloface_8n.onnx.prototxt'

WEIGHT_FACE_LANDMARKER_PATH = '2dfan4.onnx'
MODEL_FACE_LANDMARKER_PATH = '2dfan4.onnx.prototxt'

WEIGHT_FACE_RECOGNIZER_PATH = 'arcface_w600k_r50.onnx'
MODEL_FACE_RECOGNIZER_PATH = 'arcface_w600k_r50.onnx.prototxt'

WEIGHT_GENDER_AGE_PATH = 'gender_age.onnx'
MODEL_GENDER_AGE_PATH = 'gender_age.onnx.prototxt'

WEIGHT_FACE_SWAPPER_PATH = 'inswapper_128.onnx'
MODEL_FACE_SWAPPER_PATH = 'inswapper_128.onnx.prototxt'

FACE_DETECTOR_SCORE = 0.5
OUTPUT_IMAGE_QUALITY = 80
REFERENCE_FACE_DISTANCE = 0.6


def get_model_matrix(model_path):
    model = onnx.load(model_path)
    return numpy_helper.to_array(model.graph.initializer[-1])

def conditional_append_reference_faces(source_img_paths, target_img_path, nets):
    source_frames = read_static_images(source_img_paths)
    source_face = get_average_face(source_frames)

    reference_frame = read_static_image(target_img_path)
    reference_face = get_one_face(reference_frame, nets, FACE_DETECTOR_SCORE)
    append_reference_face('origin', reference_face)
    if source_face and reference_face:
        abstract_reference_frame = get_reference_frame(source_face, reference_face, reference_frame, nets)
        if np.any(abstract_reference_frame):
            reference_frame = abstract_reference_frame
            reference_face = get_one_face(reference_frame, nets, FACE_DETECTOR_SCORE)
            append_reference_face('face_swapper', reference_face)

def process_image(source_img_paths, target_img_path, output_img_path, nets):
    shutil.copy2(target_img_path, output_img_path)
    # process frame
    _process_image(source_img_paths, output_img_path, output_img_path, REFERENCE_FACE_DISTANCE, nets, FACE_DETECTOR_SCORE)
    post_process()
    # compress image
    compress_image(output_img_path, OUTPUT_IMAGE_QUALITY)


def main():
    dic_model = {
        'face_detector': (WEIGHT_FACE_DETECTOR_PATH, MODEL_FACE_DETECTOR_PATH),
        'face_landmarker': (WEIGHT_FACE_LANDMARKER_PATH, MODEL_FACE_LANDMARKER_PATH),
        'face_recognizer': (WEIGHT_FACE_RECOGNIZER_PATH, MODEL_FACE_RECOGNIZER_PATH),
        'gender_age': (WEIGHT_GENDER_AGE_PATH, MODEL_GENDER_AGE_PATH),
        'face_swapper': (WEIGHT_FACE_SWAPPER_PATH, MODEL_FACE_SWAPPER_PATH),
    }

    if False:
        nets = {k: ailia.Net(v[1], v[0], env_id=args.env_id) for k, v in dic_model.items()}
    else:
        import onnxruntime
        cuda = False #0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']

        nets = {k: onnxruntime.InferenceSession(v[0], providers=providers) for k, v in dic_model.items()}

    nets['is_onnx'] = True
    nets['model_matrix'] = get_model_matrix(dic_model['face_swapper'][0])

    source_img_paths = ['/Users/nathan/Documents/images/docomo_megatest/0006.jpg',]
    pre_process(source_img_paths, nets, FACE_DETECTOR_SCORE)

    target_img_path = '/Users/nathan/Documents/images/docomo_megatest/0002.jpg'
    conditional_append_reference_faces(source_img_paths, target_img_path, nets)

    output_img_path = '/Users/nathan/Desktop/trash/facefusion/output.jpg'
    process_image(source_img_paths, target_img_path, output_img_path, nets)


if __name__ == '__main__':
    main()
