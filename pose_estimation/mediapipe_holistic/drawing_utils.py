import math

import cv2
import matplotlib.pyplot as plt

from landmark_const import HAND_CONNECTION, FACEMESH_TESSELATION

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5

POSE_CONNECTIONS = {
    (15, 21), (16, 20), (18, 20), (3, 7), (14, 16),
    (23, 25), (28, 30), (11, 23), (27, 31), (6, 8),
    (15, 17), (24, 26), (16, 22), (4, 5), (5, 6),
    (29, 31), (12, 24), (23, 24), (0, 1), (9, 10),
    (1, 2), (0, 4), (11, 13), (30, 32), (28, 32),
    (15, 19), (16, 18), (25, 27), (26, 28), (12, 14),
    (17, 19), (2, 3), (11, 12), (27, 29), (13, 15)
}

LANDMARK_CENTER = (0,)
LANDMARK_LEFT = (1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31)

WHITE_COLOR = (224, 224, 224)
GRAY_COLOR = (128, 128, 128)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float,
        image_width: int, image_height: int):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) \
               and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    return x_px, y_px


def draw_landmarks(
        image,
        landmark_list):
    """Draws the landmarks and the connections on the image.

    Args:
      image: A three channel BGR image represented as numpy ndarray.
      landmark_list: A normalized landmark list proto message to be annotated on the image.
    """
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}

    for idx, landmark in enumerate(landmark_list):
        if landmark.visibility < _VISIBILITY_THRESHOLD or \
                landmark.presence < _PRESENCE_THRESHOLD:
            continue
        landmark_px = _normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    thickness = 2
    circle_radius = 2

    # Draws the connections if the start and end landmarks are both visible.
    for connection in POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            cv2.line(
                image, idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx],
                WHITE_COLOR, thickness)

    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    for idx, landmark_px in idx_to_coordinates.items():
        color = WHITE_COLOR
        if idx in LANDMARK_LEFT:
            color = (0, 138, 255)
        elif idx not in LANDMARK_CENTER:
            color = (231, 217, 0)

        # White circle border
        circle_border_radius = max(
            circle_radius + 1, int(circle_radius * 1.2))
        cv2.circle(
            image, landmark_px, circle_border_radius, WHITE_COLOR, thickness)
        # Fill color into the circle
        cv2.circle(
            image, landmark_px, circle_radius, color, thickness)

    return


def draw_face_landmarks(
        image,
        landmark_list):
    """Draws the landmarks and the connections on the image.

    Args:
      image: A three channel BGR image represented as numpy ndarray.
      landmark_list: A normalized landmark list proto message to be annotated on the image.
    """
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}

    for idx, landmark in enumerate(landmark_list):
        landmark_px = _normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    thickness = 1

    # Draws the connections if the start and end landmarks are both visible.
    for connection in FACEMESH_TESSELATION:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            cv2.line(
                image, idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx],
                GRAY_COLOR, thickness)

    return


def draw_hand_landmarks(
        image,
        landmark_list):
    """Draws the landmarks and the connections on the image.

    Args:
      image: A three channel BGR image represented as numpy ndarray.
      landmark_list: A normalized landmark list proto message to be annotated on the image.
    """
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}

    for idx, landmark in enumerate(landmark_list):
        landmark_px = _normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    # Draws the connections if the start and end landmarks are both visible.
    for connection in HAND_CONNECTION:
        color = GRAY_COLOR
        thickness = 3
        if connection in ((1, 2), (2, 3), (3, 4)):
            color = (180, 229, 255)
            thickness = 2
        elif connection in ((5, 6), (6, 7), (7, 8)):
            color = (128, 64, 128)
            thickness = 2
        elif connection in ((9, 10), (10, 11), (11, 12)):
            color = (0, 204, 255)
            thickness = 2
        elif connection in ((13, 14), (14, 15), (15, 16)):
            color = (48, 255, 48)
            thickness = 2
        elif connection in ((17, 18), (18, 19), (19, 20)):
            color = (192, 101, 21)
            thickness = 2
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            cv2.line(
                image, idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx],
                color, thickness)

    for idx, landmark_px in idx_to_coordinates.items():
        color = (48, 48, 255)
        thickness = -1
        circle_radius = 5
        if idx in (2, 3, 4):
            # THUMB_MCP, THUMB_IP, THUMB_TIP
            color = (180, 229, 255)
        elif idx in (6, 7, 8):
            # INDEX_FINGER_PIP, INDEX_FINGER_DIP, INDEX_FINGER_TIP
            color = (128, 64, 128)
        elif idx in (10, 11, 12):
            # MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP
            color = (0, 204, 255)
        elif idx in (14, 15, 16):
            # RING_FINGER_PIP, RING_FINGER_DIP, RING_FINGER_TIP
            color = (48, 255, 48)
        elif idx in (18, 19, 20):
            # PINKY_PIP, PINKY_DIP, PINKY_TIP
            color = (192, 101, 21)

        # White circle border
        circle_border_radius = max(
            circle_radius + 1, int(circle_radius * 1.2))
        cv2.circle(
            image, landmark_px, circle_border_radius, WHITE_COLOR, thickness)
        # Fill color into the circle
        cv2.circle(
            image, landmark_px, circle_radius, color, thickness)

    return


def plot_landmarks(
        landmark_list,
        elevation=10,
        azimuth=10):
    """Plot the landmarks and the connections in matplotlib 3d.

    Args:
      landmark_list: A normalized landmark list proto message to be plotted.
      elevation: The elevation from which to view the plot.
      azimuth: the azimuth angle to rotate the plot.
    """
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)

    color = tuple(v / 255. for v in RED_COLOR[::-1])
    thickness = 5

    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list):
        if landmark.visibility < _VISIBILITY_THRESHOLD:
            continue
        ax.scatter3D(
            xs=[-landmark.z],
            ys=[landmark.x],
            zs=[-landmark.y],
            color=color,
            linewidth=thickness)
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)

    color = tuple(v / 255. for v in BLACK_COLOR[::-1])

    # Draws the connections if the start and end landmarks are both visible.
    for connection in POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
            landmark_pair = [
                plotted_landmarks[start_idx], plotted_landmarks[end_idx]
            ]
            ax.plot3D(
                xs=[landmark_pair[0][0], landmark_pair[1][0]],
                ys=[landmark_pair[0][1], landmark_pair[1][1]],
                zs=[landmark_pair[0][2], landmark_pair[1][2]],
                color=color,
                linewidth=thickness)

    plt.show()
