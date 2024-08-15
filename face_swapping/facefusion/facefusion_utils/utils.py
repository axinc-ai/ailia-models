from collections import namedtuple
import cv2
import numpy as np

FACE_MASK_BLUR = 0.3
FACE_MASK_PADDING = (0, 0, 0, 0)

FACE_DETECTOR_SIZE = '640x640'

TEMPLATES = {'arcface_112_v2': np.array([[ 0.34191607, 0.46157411 ],
                                         [ 0.65653393, 0.45983393 ],
                                         [ 0.50022500, 0.64050536 ],
                                         [ 0.37097589, 0.82469196 ],
                                         [ 0.63151696, 0.82325089 ]]),
             'arcface_128_v2': np.array([[ 0.36167656, 0.40387734 ],
                                         [ 0.63696719, 0.40235469 ],
                                         [ 0.50019687, 0.56044219 ],
                                         [ 0.38710391, 0.72160547 ],
                                         [ 0.61507734, 0.72034453 ]]),
             'ffhq_512': np.array([[ 0.37691676, 0.46864664 ],
                                   [ 0.62285697, 0.46912813 ],
                                   [ 0.50123859, 0.61331904 ],
                                   [ 0.39308822, 0.72541100 ],
                                   [ 0.61150205, 0.72490465 ]])}

Face = namedtuple('Face',
                  ['bounding_box',
                   'landmark',
                   'score',
                   'embedding',
                   'normed_embedding'])


def read_static_image(image_path):
    if isinstance(image_path, str):
        return cv2.imread(image_path)
    else:
        return image_path  # already data

def read_static_images(image_paths):
    frames = []
    if image_paths:
        for image_path in image_paths:
            frames.append(read_static_image(image_path))
    return frames

def write_image(image_path, frame):
    if image_path:
        return cv2.imwrite(image_path, frame)

def unpack_resolution(resolution):
    width, height = map(int, resolution.split('x'))
    return width, height

def resize_frame_resolution(vision_frame, max_width, max_height):
    height, width = vision_frame.shape[:2]

    if height > max_height or width > max_width:
        scale = min(max_height / height, max_width / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(vision_frame, (new_width, new_height))
    return vision_frame

def apply_nms(bounding_box_list, iou_threshold):
    keep_indices = []
    dimension_list = np.reshape(bounding_box_list, (-1, 4))
    x1 = dimension_list[:, 0]
    y1 = dimension_list[:, 1]
    x2 = dimension_list[:, 2]
    y2 = dimension_list[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.arange(len(bounding_box_list))
    while indices.size > 0:
        index = indices[0]
        remain_indices = indices[1:]
        keep_indices.append(index)
        xx1 = np.maximum(x1[index], x1[remain_indices])
        yy1 = np.maximum(y1[index], y1[remain_indices])
        xx2 = np.minimum(x2[index], x2[remain_indices])
        yy2 = np.minimum(y2[index], y2[remain_indices])
        width = np.maximum(0, xx2 - xx1 + 1)
        height = np.maximum(0, yy2 - yy1 + 1)
        iou = width * height / (areas[index] + areas[remain_indices] - width * height)
        indices = indices[np.where(iou <= iou_threshold)[0] + 1]
    return keep_indices

def warp_face_by_translation(temp_vision_frame, translation, scale, crop_size):
    affine_matrix = np.array([[ scale, 0, translation[0] ], [ 0, scale, translation[1] ]])
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size)
    return crop_vision_frame, affine_matrix

def convert_face_landmark_68_to_5(landmark_68):
    left_eye = np.mean(landmark_68[36:42], axis = 0)
    right_eye = np.mean(landmark_68[42:48], axis = 0)
    nose = landmark_68[30]
    left_mouth_end = landmark_68[48]
    right_mouth_end = landmark_68[54]
    face_landmark_5 = np.array([left_eye, right_eye, nose, left_mouth_end, right_mouth_end])
    return face_landmark_5

def warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, template, crop_size):
    normed_template = TEMPLATES.get(template) * crop_size
    affine_matrix = cv2.estimateAffinePartial2D(face_landmark_5, normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
    return crop_vision_frame, affine_matrix

def get_first(__list__):
    return next(iter(__list__), None)

def paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix):
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    temp_size = temp_vision_frame.shape[:2][::-1]
    inverse_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_size).clip(0, 1)
    inverse_vision_frame = cv2.warpAffine(crop_vision_frame, inverse_matrix, temp_size, borderMode = cv2.BORDER_REPLICATE)
    paste_vision_frame = temp_vision_frame.copy()
    paste_vision_frame[:, :, 0] = inverse_mask * inverse_vision_frame[:, :, 0] + (1 - inverse_mask) * temp_vision_frame[:, :, 0]
    paste_vision_frame[:, :, 1] = inverse_mask * inverse_vision_frame[:, :, 1] + (1 - inverse_mask) * temp_vision_frame[:, :, 1]
    paste_vision_frame[:, :, 2] = inverse_mask * inverse_vision_frame[:, :, 2] + (1 - inverse_mask) * temp_vision_frame[:, :, 2]
    return paste_vision_frame
