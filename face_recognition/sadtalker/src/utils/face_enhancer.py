import os
import sys
import torch 

from gfpgan import GFPGANer

from tqdm import tqdm

from src.utils.videoio import load_video_to_cv2

import cv2

import onnxruntime
import numpy as np

sys.path.append('../../util')
sys.path.append('../../face_restoration/gfpgan')
from face_restoration import get_face_landmarks_5, align_warp_face, get_inverse_affine, paste_faces_to_image


class GeneratorWithLen(object):
    """ From https://stackoverflow.com/a/7460929 """

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen

def enhancer_list(images, method='gfpgan', bg_upsampler='realesrgan'):
    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    return list(gen)

def enhancer_generator_with_len(images, method='gfpgan', bg_upsampler='realesrgan', retinaface_net=None, gfpgan_net=None):
    """ Provide a generator with a __len__ method so that it can passed to functions that
    call len()"""

    if os.path.isfile(images): # handle video to images
        # TODO: Create a generator version of load_video_to_cv2
        images = load_video_to_cv2(images)

    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler, retinaface_net=retinaface_net, gfpgan_net=gfpgan_net)
    gen_with_len = GeneratorWithLen(gen, len(images))
    return gen_with_len

def preprocess(img):
    img = img[:, :, ::-1]  # BGR -> RGB
    img = img / 127.5 - 1.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img

def post_processing(pred):
    img = pred[0]
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img[:, :, ::-1]  # RGB -> BGR

    img = np.clip(img, -1, 1)
    img = (img + 1) * 127.5
    img = img.astype(np.uint8)

    return img

def enhancer_generator_no_len(images, method='gfpgan', bg_upsampler='realesrgan', retinaface_net=None, gfpgan_net=None):
    """ Provide a generator function so that all of the enhanced images don't need
    to be stored in memory at the same time. This can save tons of RAM compared to
    the enhancer function. """

    print('face enhancer....')
    if not isinstance(images, list) and os.path.isfile(images): # handle video to images
        images = load_video_to_cv2(images)

    # ------------------------ set up GFPGAN restorer ------------------------
    if  method == 'gfpgan':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif method == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    elif method == 'codeformer': # TODO:
        arch = 'CodeFormer'
        channel_multiplier = 2
        model_name = 'CodeFormer'
        url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    else:
        raise ValueError(f'Wrong model version {method}.')


    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # determine model paths
    model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    
    if not os.path.isfile(model_path):
        model_path = os.path.join('checkpoints', model_name + '.pth')
    
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    for idx in tqdm(range(len(images)), 'Face Enhancer:'):
        
        img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)

        det_faces, all_landmarks_5 = get_face_landmarks_5(retinaface_net, img, eye_dist_threshold=5)
        cropped_faces, affine_matrices = align_warp_face(img, all_landmarks_5)

        restored_faces = []
        for cropped_face in cropped_faces:
            x = preprocess(cropped_face)

            # feedforward
            output = gfpgan_net.predict([x])
            pred = output[0]

            restored_face = post_processing(pred)
            restored_faces.append(restored_face)

        upscale = 2

        h, w = img.shape[:2]
        h_up, w_up = int(h * upscale), int(w * upscale)
        img = cv2.resize(img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        
        inverse_affine_matrices = get_inverse_affine(affine_matrices, upscale_factor=upscale)
        r_img = paste_faces_to_image(img, restored_faces, inverse_affine_matrices, upscale_factor=upscale)

        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        yield r_img
