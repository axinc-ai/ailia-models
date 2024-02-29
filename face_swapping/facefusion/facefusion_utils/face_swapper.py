import numpy as np

from .face_analyser import get_one_face, get_average_face, find_similar_faces
from .utils import read_static_images, read_static_image, write_image, warp_face_by_face_landmark_5, paste_back
from .face_masker import create_static_box_mask
from .face_store import get_reference_faces, clear_reference_faces, clear_static_faces

FACE_MASK_BLUR = 0.3
FACE_MASK_PADDING = (0, 0, 0, 0)

INSWAPPER_128_MODEL_MEAN = [0.0, 0.0, 0.0]
INSWAPPER_128_MODEL_STD = [1.0, 1.0, 1.0]


def prepare_crop_frame(crop_vision_frame):
    crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
    crop_vision_frame = (crop_vision_frame - INSWAPPER_128_MODEL_MEAN) / INSWAPPER_128_MODEL_STD
    crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
    crop_vision_frame = np.expand_dims(crop_vision_frame, axis = 0).astype(np.float32)
    return crop_vision_frame

def prepare_source_embedding(source_face, model_matrix):
    source_embedding = source_face.embedding.reshape((1, -1))
    source_embedding = np.dot(source_embedding, model_matrix) / np.linalg.norm(source_embedding)

    return source_embedding

def apply_swap(source_face, crop_vision_frame, nets):
    frame_processor = nets['face_swapper']
    frame_processor_inputs = {
        'source': prepare_source_embedding(source_face, nets['model_matrix']),
        'target': crop_vision_frame
    }

    if nets['is_onnx']:
        crop_vision_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
    else:
        crop_vision_frame = frame_processor.predict(frame_processor_inputs)[0][0]

    return crop_vision_frame

def normalize_crop_frame(crop_vision_frame):
    crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
    crop_vision_frame = (crop_vision_frame * 255.0).round()
    crop_vision_frame = crop_vision_frame[:, :, ::-1]
    return crop_vision_frame

def swap_face(source_face, target_face, temp_vision_frame, nets):
    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark['5/68'], 'arcface_128_v2', (128, 128))
    crop_mask_list = []

    box_mask = create_static_box_mask(crop_vision_frame.shape[:2][::-1], FACE_MASK_BLUR, FACE_MASK_PADDING)
    crop_mask_list.append(box_mask)

    crop_vision_frame = prepare_crop_frame(crop_vision_frame)
    crop_vision_frame = apply_swap(source_face, crop_vision_frame, nets)
    crop_vision_frame = normalize_crop_frame(crop_vision_frame)

    crop_mask = np.minimum.reduce(crop_mask_list).clip(0, 1)
    temp_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
    return temp_vision_frame

def get_reference_frame(source_face, target_face, temp_vision_frame, nets):
    return swap_face(source_face, target_face, temp_vision_frame, nets)

def pre_process(source_img_paths, nets, face_detector_score):
    source_frames = read_static_images(source_img_paths)

    for source_frame in source_frames:
        get_one_face(source_frame, nets, face_detector_score)

    return True

def process_frame(inputs, reference_face_distance, nets, face_detector_score):
    reference_faces = inputs['reference_faces']
    source_face = inputs['source_face']
    target_vision_frame = inputs['target_vision_frame']

    similar_faces = find_similar_faces(reference_faces, target_vision_frame, reference_face_distance, nets, face_detector_score)
    if similar_faces:
        for similar_face in similar_faces:
            target_vision_frame = swap_face(source_face, similar_face, target_vision_frame, nets)

    return target_vision_frame

def process_image(source_paths, target_path, output_path, reference_face_distance, nets, face_detector_score):
    reference_faces = get_reference_faces()
    source_frames = read_static_images(source_paths)
    source_face = get_average_face(source_frames, nets, face_detector_score)
    target_vision_frame = read_static_image(target_path)
    result_frame = process_frame({'reference_faces': reference_faces,
                                    'source_face': source_face,
                                    'target_vision_frame': target_vision_frame},
                                reference_face_distance,
                                nets,
                                face_detector_score)
    write_image(output_path, result_frame)

def post_process():
    clear_static_faces()
    clear_reference_faces()
