import sys
import os

import cv2
import numpy as np
import pickle
from pathlib import Path

# import original modules
sys.path.append('../../util')
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
NORM_MEAN = [123.675, 116.28, 103.53]
NORM_STD = [58.395, 57.12, 57.375]


def preprocess(img):
    # scale
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # normalize
    img = img.astype(np.float32)
    mean = np.array(NORM_MEAN)
    std = np.array(NORM_STD)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    return img

def process_embeds(gallery, gallery_imgs, gallery_embeds, model):
    logger.info('Exploring the gallery... (it may take a while)')
    modified = False

    # If some images have been removed from the gallery
    if len(gallery_imgs) <= len(gallery_embeds):
        removed_keys = set(gallery_embeds) - set(gallery_imgs)
        for key in removed_keys:
            modified = True
            del gallery_embeds[key]

    # If some images have been added or replaced from the gallery
    if len(gallery_imgs) >= len(gallery_embeds):
        added_keys = set(gallery_imgs) - set(gallery_embeds)
        for key in added_keys:
            modified = True
            # prepare input data
            img = load_image(os.path.join(gallery, key))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = preprocess(img)
            # inference
            embed = model.predict(img)
            gallery_embeds[key] = embed

            if len(gallery_embeds)+1 % 500 == 0: 
                print(f"{len(gallery_embeds)}/{len(gallery_imgs)}")
    
    # Saves the new embeds dict if modified
    if modified:
        folder = os.path.join(gallery, 'gallery_embeds.pkl')
        with open(folder, 'wb') as file:
            pickle.dump(gallery_embeds, file)
            logger.info(f'Gallery embeds saved at : {folder}')


def process_gallery(gallery, net):
    
    gallery_imgs_path = os.path.join(gallery, 'gallery_imgs.txt')
    gallery_embeds_path = os.path.join(gallery, 'gallery_embeds.pkl')

    generate_images_filename_txt(gallery)
    gallery_imgs = open(gallery_imgs_path, 'r').read().splitlines()

    if os.path.isfile(gallery_embeds_path):
        file = open(gallery_embeds_path, "rb")
        gallery_embeds = pickle.load(file)
        file.close()
    else:
        gallery_embeds = {}

    process_embeds(gallery, gallery_imgs, gallery_embeds, net)

    return gallery_imgs, gallery_embeds
    

def generate_images_filename_txt(root):
    # looks recursively for .jpg/.JPG or .png/.PNG files from the root directory
    paths = list(Path(root).rglob("*.[jJ|pP][pP|nN][gG]"))
    # relative paths from the root directory
    filenames = [path.relative_to(root) for path in paths]
    folder = os.path.join(root, 'gallery_imgs.txt')
    with open(folder, 'w') as f:
        for filename in sorted(filenames):
            f.write("%s\n" % filename)
    logger.info(f'Gallery image filenames saved at : {folder}')