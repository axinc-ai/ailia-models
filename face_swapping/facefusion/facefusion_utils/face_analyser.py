import cv2
import numpy as np

from .face_store import get_static_faces, set_static_faces
from .utils import unpack_resolution, resize_frame_resolution, warp_face_by_translation, get_first, \
    warp_face_by_face_landmark_5, apply_nms, convert_face_landmark_68_to_5, Face, FACE_DETECTOR_SIZE

# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)


def prepare_detect_frame(temp_vision_frame, face_detector_size):
    face_detector_width, face_detector_height = unpack_resolution(face_detector_size)
    detect_vision_frame = np.zeros((face_detector_height, face_detector_width, 3))
    detect_vision_frame[:temp_vision_frame.shape[0], :temp_vision_frame.shape[1], :] = temp_vision_frame
    detect_vision_frame = (detect_vision_frame - 127.5) / 128.0
    detect_vision_frame = np.expand_dims(detect_vision_frame.transpose(2, 0, 1), axis = 0).astype(np.float32)
    return detect_vision_frame

def detect_with_yoloface(vision_frame, face_detector, face_detector_size, min_score, is_onnx):
    face_detector_width, face_detector_height = unpack_resolution(face_detector_size)
    temp_vision_frame = resize_frame_resolution(vision_frame, face_detector_width, face_detector_height)
    ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
    ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
    bounding_box_list = []
    face_landmark5_list = []
    score_list = []

    if is_onnx:
        detections = face_detector.run(None, {
            face_detector.get_inputs()[0].name: prepare_detect_frame(temp_vision_frame, face_detector_size)
        })
    else:
        detections = face_detector.predict([prepare_detect_frame(temp_vision_frame, face_detector_size)])

    detections = np.squeeze(detections).T
    bounding_box_raw, score_raw, face_landmark_5_raw = np.split(detections, [ 4, 5 ], axis = 1)
    keep_indices = np.where(score_raw > min_score)[0]
    if keep_indices.any():
        bounding_box_raw, face_landmark_5_raw, score_raw = bounding_box_raw[keep_indices], face_landmark_5_raw[keep_indices], score_raw[keep_indices]
        for bounding_box in bounding_box_raw:
            bounding_box_list.append(np.array(
            [
                (bounding_box[0] - bounding_box[2] / 2) * ratio_width,
                (bounding_box[1] - bounding_box[3] / 2) * ratio_height,
                (bounding_box[0] + bounding_box[2] / 2) * ratio_width,
                (bounding_box[1] + bounding_box[3] / 2) * ratio_height
            ]))
        face_landmark_5_raw[:, 0::3] = (face_landmark_5_raw[:, 0::3]) * ratio_width
        face_landmark_5_raw[:, 1::3] = (face_landmark_5_raw[:, 1::3]) * ratio_height
        for face_landmark_5 in face_landmark_5_raw:
            face_landmark5_list.append(np.array(face_landmark_5.reshape(-1, 3)[:, :2]))
        score_list = score_raw.ravel().tolist()
    return bounding_box_list, face_landmark5_list, score_list

def detect_face_landmark_68(temp_vision_frame, face_landmarker, bounding_box, is_onnx):
    scale = 195 / np.subtract(bounding_box[2:], bounding_box[:2]).max()
    translation = (256 - np.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
    crop_vision_frame, affine_matrix = warp_face_by_translation(temp_vision_frame, translation, scale, (256, 256))
    crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(np.float32) / 255.0

    if is_onnx:
        face_landmark_68 = face_landmarker.run(None, {
            face_landmarker.get_inputs()[0].name: [ crop_vision_frame ]
        })[0]
    else:
        face_landmark_68 = face_landmarker.predict([np.expand_dims(crop_vision_frame, axis=0)])[0]

    face_landmark_68 = face_landmark_68[:, :, :2][0] / 64
    face_landmark_68 = face_landmark_68.reshape(1, -1, 2) * 256
    face_landmark_68 = cv2.transform(face_landmark_68, cv2.invertAffineTransform(affine_matrix))
    face_landmark_68 = face_landmark_68.reshape(-1, 2)
    return face_landmark_68

def calc_embedding(temp_vision_frame, face_recognizer, face_landmark_5, is_onnx):
    crop_vision_frame, _ = warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, 'arcface_112_v2', (112, 112))
    crop_vision_frame = crop_vision_frame / 127.5 - 1
    crop_vision_frame = crop_vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    crop_vision_frame = np.expand_dims(crop_vision_frame, axis = 0)

    if is_onnx:
        embedding = face_recognizer.run(None, {
            face_recognizer.get_inputs()[0].name: crop_vision_frame
        })[0]
    else:
        embedding = face_recognizer.predict([crop_vision_frame])[0]

    embedding = embedding.ravel()
    normed_embedding = embedding / np.linalg.norm(embedding)
    return embedding, normed_embedding

def create_faces(vision_frame, nets, bounding_box_list, face_landmark5_list, score_list, min_score):
    faces = []
    if min_score > 0:
        sort_indices = np.argsort(-np.array(score_list))
        bounding_box_list = [ bounding_box_list[index] for index in sort_indices ]
        face_landmark5_list = [ face_landmark5_list[index] for index in sort_indices ]
        score_list = [ score_list[index] for index in sort_indices ]
        keep_indices = apply_nms(bounding_box_list, 0.4)
        for index in keep_indices:
            bounding_box = bounding_box_list[index]
            face_landmark_68 = detect_face_landmark_68(vision_frame, nets['face_landmarker'], bounding_box, nets['is_onnx'])
            landmark = {
                '5': face_landmark5_list[index],
                '5/68': convert_face_landmark_68_to_5(face_landmark_68),
                '68': face_landmark_68
            }
            score = score_list[index]
            embedding, normed_embedding = calc_embedding(vision_frame, nets['face_recognizer'], landmark['5/68'], nets['is_onnx'])
            faces.append(Face(
                bounding_box = bounding_box,
                landmark = landmark,
                score = score,
                embedding = embedding,
                normed_embedding = normed_embedding
            ))
    return faces

def sort_by_order(faces, order):
    if order == 'left-right':
        return sorted(faces, key = lambda face: face.bounding_box[0])
    if order == 'right-left':
        return sorted(faces, key = lambda face: face.bounding_box[0], reverse = True)
    if order == 'top-bottom':
        return sorted(faces, key = lambda face: face.bounding_box[1])
    if order == 'bottom-top':
        return sorted(faces, key = lambda face: face.bounding_box[1], reverse = True)
    if order == 'small-large':
        return sorted(faces, key = lambda face: (face.bounding_box[2] - face.bounding_box[0]) * (face.bounding_box[3] - face.bounding_box[1]))
    if order == 'large-small':
        return sorted(faces, key = lambda face: (face.bounding_box[2] - face.bounding_box[0]) * (face.bounding_box[3] - face.bounding_box[1]), reverse = True)
    if order == 'best-worst':
        return sorted(faces, key = lambda face: face.score, reverse = True)
    if order == 'worst-best':
        return sorted(faces, key = lambda face: face.score)
    return faces

def get_many_faces(vision_frame, nets, min_score):
    faces = []
    try:
        faces_cache = get_static_faces(vision_frame)
        if faces_cache:
            faces = faces_cache
        else:
            bounding_box_list, face_landmark5_list, score_list = detect_with_yoloface(vision_frame, nets['face_detector'], FACE_DETECTOR_SIZE, min_score, nets['is_onnx'])
            faces = create_faces(vision_frame, nets, bounding_box_list, face_landmark5_list, score_list, min_score)
            if faces:
                set_static_faces(vision_frame, faces)
        faces = sort_by_order(faces, 'left-right')
    except (AttributeError, ValueError) as e:
        logger.error(e)
        pass
    return faces

def get_one_face(vision_frame, nets, min_score=0.5, position=0):
    many_faces = get_many_faces(vision_frame, nets, min_score)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None

def get_average_face(vision_frames, nets, face_detector_score, position=0):
    average_face = None
    faces = []
    embedding_list = []
    normed_embedding_list = []

    for vision_frame in vision_frames:
        face = get_one_face(vision_frame, nets, face_detector_score, position)
        if face:
            faces.append(face)
            embedding_list.append(face.embedding)
            normed_embedding_list.append(face.normed_embedding)
    if faces:
        first_face = get_first(faces)
        average_face = Face(
            bounding_box = first_face.bounding_box,
            landmark = first_face.landmark,
            score = first_face.score,
            embedding = np.mean(embedding_list, axis = 0),
            normed_embedding = np.mean(normed_embedding_list, axis = 0)
        )
    return average_face

def compare_faces(face, reference_face, face_distance):
    current_face_distance = calc_face_distance(face, reference_face)
    return current_face_distance < face_distance

def calc_face_distance(face, reference_face):
    if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
        return 1 - np.dot(face.normed_embedding, reference_face.normed_embedding)
    return 0

def find_similar_faces(reference_faces, vision_frame, face_distance, nets, face_detector_score):
    similar_faces = []
    many_faces = get_many_faces(vision_frame, nets, face_detector_score)

    if reference_faces:
        for reference_set in reference_faces:
            if not similar_faces:
                for reference_face in reference_faces[reference_set]:
                    for face in many_faces:
                        if compare_faces(face, reference_face, face_distance):
                            similar_faces.append(face)
    return similar_faces
