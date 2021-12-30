"""Methods for drawing a bounding box on an image."""
import cv2
import numpy as np

import objectron.dataset.box as Box

_LINE_TICKNESS = 10
_POINT_RADIUS = 10
_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 255),
]


def draw_annotation_on_image(image,
                             object_annotations,
                             num_keypoints):
    """Draw annotation on the image."""
    # The object annotation is a list of 3x1 keypoints for all the annotated
    # objects. The objects can have a varying number of keypoints. First we split
    # the list according to the number of keypoints for each object. This
    # also leaves an empty array at the end of the list.
    keypoints = np.split(object_annotations, np.array(np.cumsum(num_keypoints)))
    keypoints = [points.reshape(-1, 3) for points in keypoints]
    h, w, _ = image.shape
    num_objects = len(num_keypoints)
    # The keypoints are [x, y, d] where `x` and `y` are normalized (`uv`-system)\
    # and `d` is the metric distance from the center of the camera. Convert them
    # keypoint's `xy` value to pixel.
    keypoints = [
        np.multiply(keypoint, np.asarray([w, h, 1.], np.float32)).astype(int)
        for keypoint in keypoints
    ]

    def draw_face(object_id, face, color):
        start = keypoints[object_id][face[0], :]
        end = keypoints[object_id][face[2], :]
        cv2.line(image, (start[0], start[1]), (end[0], end[1]), color,
                 _LINE_TICKNESS)
        start = keypoints[object_id][face[1], :]
        end = keypoints[object_id][face[3], :]
        cv2.line(image, (start[0], start[1]), (end[0], end[1]), color,
                 _LINE_TICKNESS)

    for object_id in range(num_objects):
        num_keypoint = num_keypoints[object_id]
        edges = Box.EDGES
        hidden = [False] * Box.NUM_KEYPOINTS
        draw_face(object_id, Box.FACES[Box.FRONT_FACE_ID], _COLORS[7])
        draw_face(object_id, Box.FACES[Box.TOP_FACE_ID], _COLORS[8])

        for kp_id in range(num_keypoint):
            kp_pixel = keypoints[object_id][kp_id, :]
            # If a keypoint is hidden (e.g. a subset of a larger skeleton family) do
            # not visualize it.
            if not hidden[kp_id]:
                cv2.circle(image, (kp_pixel[0], kp_pixel[1]), _POINT_RADIUS,
                           _COLORS[object_id % len(_COLORS)], -1)

        for edge in edges:
            # This if statement is for backward compatibility, where we might later
            # add more edges/keypoints to the skeletons.
            if edge[0] < num_keypoint and edge[1] < num_keypoint:
                start_kp = keypoints[object_id][edge[0], :]
                end_kp = keypoints[object_id][edge[1], :]
                if not hidden[edge[0]] and not hidden[edge[1]]:
                    cv2.line(image, (start_kp[0], start_kp[1]), (end_kp[0], end_kp[1]),
                             _COLORS[object_id % len(_COLORS)], _LINE_TICKNESS)
    return image
