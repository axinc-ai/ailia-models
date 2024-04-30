import cv2
import numpy as np

from .face_masker import create_static_box_mask
from .utils import warp_face_by_face_landmark_5, paste_back, FACE_MASK_BLUR, FACE_MASK_PADDING
from .face_store import get_reference_faces
from .face_analyser import find_similar_faces

FACE_ENHANCER_BLEND = 80


def prepare_crop_frame(crop_vision_frame):
    crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
    crop_vision_frame = (crop_vision_frame - 0.5) / 0.5
    crop_vision_frame = np.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis = 0).astype(np.float32)
    return crop_vision_frame

def apply_enhance(crop_vision_frame, nets):
    frame_processor = nets['face_enhancer']
    frame_processor_inputs = {
        'input': crop_vision_frame
    }

    if nets['is_onnx']:
        crop_vision_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
    else:
        crop_vision_frame = frame_processor.predict(frame_processor_inputs)[0][0]

    return crop_vision_frame

def normalize_crop_frame(crop_vision_frame):
    crop_vision_frame = np.clip(crop_vision_frame, -1, 1)
    crop_vision_frame = (crop_vision_frame + 1) / 2
    crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
    crop_vision_frame = (crop_vision_frame * 255.0).round()
    crop_vision_frame = crop_vision_frame.astype(np.uint8)[:, :, ::-1]
    return crop_vision_frame

def blend_frame(temp_vision_frame, paste_vision_frame):
    face_enhancer_blend = 1 - (FACE_ENHANCER_BLEND / 100)
    temp_vision_frame = cv2.addWeighted(temp_vision_frame, face_enhancer_blend, paste_vision_frame, 1 - face_enhancer_blend, 0)
    return temp_vision_frame

def enhance_face(target_face, temp_vision_frame, nets):
    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark['5/68'], 'ffhq_512', (512, 512))
    box_mask = create_static_box_mask(crop_vision_frame.shape[:2][::-1], FACE_MASK_BLUR, FACE_MASK_PADDING)
    crop_mask_list = [ box_mask ]

    crop_vision_frame = prepare_crop_frame(crop_vision_frame)
    crop_vision_frame = apply_enhance(crop_vision_frame, nets)
    crop_vision_frame = normalize_crop_frame(crop_vision_frame)
    crop_mask = np.minimum.reduce(crop_mask_list).clip(0, 1)
    paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
    temp_vision_frame = blend_frame(temp_vision_frame, paste_vision_frame)
    return temp_vision_frame

def get_reference_frame(source_face, target_face, temp_vision_frame, nets):
    return enhance_face(target_face, temp_vision_frame, nets)

def process_frame(inputs, reference_face_distance, nets, face_detector_score):
    reference_faces = inputs['reference_faces']
    target_vision_frame = inputs['target_vision_frame']

    similar_faces = find_similar_faces(reference_faces, target_vision_frame, reference_face_distance, nets, face_detector_score)
    if similar_faces:
        for similar_face in similar_faces:
            target_vision_frame = enhance_face(similar_face, target_vision_frame, nets)

    return target_vision_frame

def process_image(source_paths, target_vision_frame, reference_face_distance, nets, face_detector_score):
    reference_faces = get_reference_faces()
    result_frame = process_frame({'reference_faces': reference_faces,
                                    'target_vision_frame': target_vision_frame},
                                reference_face_distance,
                                nets,
                                face_detector_score)

    return result_frame
