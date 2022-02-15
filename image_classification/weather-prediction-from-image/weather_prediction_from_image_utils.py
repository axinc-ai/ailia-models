import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
import PIL.ImageOps


def __find_sky_area(img_PIL):
    read_image = np.array(img_PIL, dtype=np.uint8)[:, :, ::-1]
    edges = cv2.Canny(read_image, 150, 300)

    shape = np.shape(edges)
    left = np.sum(edges[0 : shape[0] // 2, 0 : shape[1] // 2])
    right = np.sum(edges[0 : shape[0] // 2, shape[1] // 2 :])

    if right > left:
        return 0  # if right side of image includes more building etc. return 0 to define left side(0 side) is sky area
    else:
        return 1  # if left side of image includes more building etc. return 1 to define right side(1 side) is sky area


def _resize_image(img_org, base_size):
    if img_org.size[0] >= img_org.size[1]:
        sky_side = __find_sky_area(img_org)
        base_height = base_size
        wpercent = base_height / float(img_org.size[1])
        wsize = int((float(img_org.size[0]) * float(wpercent)))
        img = img_org.resize((wsize, base_height), Image.ANTIALIAS)
        if sky_side == 0:  # Left side is sky side, so keep it and crop right side
            img = img.crop(
                (0, 0, base_size, img.size[1])
            )  # Keeps sky area in image, crops from other non-sky side
        else:  # Right side is sky side, so keep it and crop left side
            img = img.crop(
                (img.size[0] - base_size, 0, img.size[0], img.size[1])
            )  # Keeps sky area in image, crops from other non-sky side
    else:
        base_width = base_size
        wpercent = base_width / float(img_org.size[0])
        hsize = int((float(img_org.size[1]) * float(wpercent)))
        img = img_org.resize((base_width, hsize), Image.ANTIALIAS)
        img = img.crop(
            (0, 0, img.size[0], base_size)
        )  # Keeps sky area in image, crops from lower part

    return img


def prepare_data_set(image, size):
    img = _resize_image(image, size)
    return _image_to_matrix(img)


def _image_to_matrix(img):
    img = PIL.ImageOps.invert(img)  # inverts it
    return np.array(img)  # converts it to array


def save_image(text, img, filename):
    annotate_text = ImageDraw.Draw(img)
    annotate_text.text((10, 10), text, fill=(255, 0, 0))
    img.save(filename)
