import numpy as np
from PIL import Image


def crop(image: Image, face, up_ratio, down_ratio, width_ratio) -> (Image, "face"):
    width, height = image.size
    face_height = face[3] - face[1]
    # face_width = face[2] - face[0]
    delta_up = up_ratio * face_height
    delta_down = down_ratio * face_height
    delta_width = width_ratio * width

    img_left = int(max(0, face[0] - delta_width))
    img_top = int(max(0, face[1] - delta_up))
    img_right = int(min(width, face[2] + delta_width))
    img_bottom = int(min(height, face[3] + delta_down))
    image = image.crop((img_left, img_top, img_right, img_bottom))
    face = [
        face[0] - img_left,
        face[1] - img_top,
        face[2] - img_left,
        face[3] - img_top,
    ]
    # face_expand = [img_left, img_top, img_right, img_bottom]
    center = {
        "x": int((img_left + img_right) / 2),
        "y": int((img_top + img_bottom) / 2),
    }
    width, height = image.size
    crop_left = img_left
    crop_top = img_top
    crop_right = img_right
    crop_bottom = img_bottom
    if width > height:
        left = int(center["x"] - height / 2)
        right = int(center["x"] + height / 2)
        if left < 0:
            left, right = 0, height
        elif right > width:
            left, right = width - height, width
        image = image.crop((left, 0, right, height))
        face = [face[0] - left, face[1], face[2] - left, face[3]]
        crop_left += left
        crop_right = crop_left + height
    elif width < height:
        top = int(center["y"] - width / 2)
        bottom = int(center["y"] + width / 2)
        if top < 0:
            top, bottom = 0, width
        elif bottom > height:
            top, bottom = height - width, height
        image = image.crop((0, top, width, bottom))
        face = [face[0], face[1] - top, face[2], face[3] - top]
        crop_top += top
        crop_bottom = crop_top + width
    crop_face = {
        "left": crop_left,
        "top": crop_top,
        "right": crop_right,
        "bottom": crop_bottom,
    }
    return image, face, crop_face


def crop_by_image_size(image: Image, face) -> (Image, "face"):
    center = {
        "x": int((face[0] + face[2]) / 2),
        "y": int((face[1] + face[3]) / 2),
    }
    width, height = image.size
    if width > height:
        left = int(center["x"] - height / 2)
        right = int(center["x"] + height / 2)
        if left < 0:
            left, right = 0, height
        elif right > width:
            left, right = width - height, width
        image = image.crop((left, 0, right, height))
        face = {
            "left": face[0] - left,
            "top": face[1],
            "right": face[2] - left,
            "bottom": face[3],
        }
    elif width < height:
        top = int(center["y"] - width / 2)
        bottom = int(center["y"] + width / 2)
        if top < 0:
            top, bottom = 0, width
        elif bottom > height:
            top, bottom = height - width, height
        image = image.crop((0, top, width, bottom))
        face = {
            "left": face[0],
            "top": face[1] - top,
            "right": face[2],
            "bottom": face[3] - top,
        }
    return image, face


def crop_from_array(image: np.array, face) -> (np.array, "face"):
    ratio = 0.20 / 0.85  # delta_size / face_size
    height, width = image.shape[:2]
    face_height = face[3] - face[1]
    # face_width = face[2] - face[0]
    delta_height = ratio * face_height
    delta_width = ratio * width

    img_left = int(max(0, face[0] - delta_width))
    img_top = int(max(0, face[1] - delta_height))
    img_right = int(min(width, face[2] + delta_width))
    img_bottom = int(min(height, face[3] + delta_height))
    image = image[img_top:img_bottom, img_left:img_right]
    face = [
        face[0] - img_left,
        face[1] - img_top,
        face[2] - img_left,
        face[3] - img_top,
    ]
    center = {
        "x": int((face[0] + face[2]) / 2),
        "y": int((face[1] + face[3]) / 2),
    }
    height, width = image.shape[:2]
    if width > height:
        left = int(center["x"] - height / 2)
        right = int(center["x"] + height / 2)
        if left < 0:
            left, right = 0, height
        elif right > width:
            left, right = width - height, width
        image = image[0:height, left:right]
        face = [
            face[0] - left,
            face[1],
            face[2] - left,
            face[3],
        ]
    elif width < height:
        top = int(center["y"] - width / 2)
        bottom = int(center["y"] + width / 2)
        if top < 0:
            top, bottom = 0, width
        elif bottom > height:
            top, bottom = height - width, height
        image = image[top:bottom, 0:width]
        face = [
            face[0],
            face[1] - top,
            face[2],
            face[3] - top,
        ]
    return image, face
