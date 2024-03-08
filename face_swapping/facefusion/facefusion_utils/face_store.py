import hashlib
import numpy as np

FACE_STORE = {
    'static_faces': {},
    'reference_faces': {}
}


def create_frame_hash(vision_frame):
    return hashlib.sha1(vision_frame.tobytes()).hexdigest() if np.any(vision_frame) else None

def get_static_faces(vision_frame):
    frame_hash = create_frame_hash(vision_frame)
    if frame_hash in FACE_STORE['static_faces']:
        return FACE_STORE['static_faces'][frame_hash]
    return None

def set_static_faces(vision_frame, faces):
    global FACE_STORE

    frame_hash = create_frame_hash(vision_frame)
    if frame_hash:
        FACE_STORE['static_faces'][frame_hash] = faces

def clear_static_faces():
    global FACE_STORE

    FACE_STORE['static_faces'] = {}

def get_reference_faces():
    if FACE_STORE['reference_faces']:
        return FACE_STORE['reference_faces']
    return None

def append_reference_face(name, face):
    global FACE_STORE

    if name not in FACE_STORE['reference_faces']:
        FACE_STORE['reference_faces'][name] = []
    FACE_STORE['reference_faces'][name].append(face)

def clear_reference_faces():
    global FACE_STORE

    FACE_STORE['reference_faces'] = {}
