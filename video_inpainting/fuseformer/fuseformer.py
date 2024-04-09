import os
import sys
import time
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger  # noqa: E402


logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_FUSEFORMER_PATH = 'fuseformer.onnx'
MODEL_FUSEFORMER_PATH = 'fuseformer.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/fuseformer/'

VIDEO_PATH = 'bmx_trees.mp4'
MASK_PATH = 'bmx_trees_mask'
SAVE_VIDEO_PATH = 'bmx_trees_out.mp4'

HEIGHT = 240
WIDTH = 432

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser('FuseFormer', None, SAVE_VIDEO_PATH)
parser.add_argument(
    '-m', '--mask',  type=str, default=MASK_PATH,
    help='Path of the mask(s) or mask folder.'
)
parser.add_argument(
    '--resize_ratio', type=float, default=1.0,
    help='Resize scale for processing video.'
)
parser.add_argument(
    '--height', type=int, default=-1,
    help='Height of the processing video.'
)
parser.add_argument(
    '--width', type=int, default=-1,
    help='Width of the processing video.'
)
parser.add_argument(
    '--step', type=int, default=10,
    help='Stride of global reference frames.'
)
parser.add_argument(
    '--num_ref', type=int, default=-1,
    help='Maximum number of global reference frames.'
)
parser.add_argument(
    '--neighbor_stride', type=int, default=5,
    help='Stride of local neighboring frames.'
)
parser.add_argument(
    '--save_fps', type=int, default=24,
    help='Frame per second. Default: 24'
)
parser.add_argument(
    '--save_frames', action='store_true',
    help='Save output frames.'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='If set, runs the inference using onnx runtime.'
)

args = update_parser(parser)


# ======================
# Helper functions
# ======================

class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")

class ToTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a numpy.ndarray of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = pic.transpose((2, 3, 0, 1))
        else:
            # handle PIL Image
            img = np.array(pic).astype(np.uint8)
            img = np.reshape(img, (pic.size[1], pic.size[0], len(pic.mode)))
            # put it from HWC to CHW format
            img = img.transpose((0, 1)).transpose((0, 2))
        img = img.astype(np.float32) / 255 if self.div else img.astype(np.float32)
        return img

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def to_tensors():
    return Compose([Stack(), ToTensor()])

def read_frame_from_videos(frame_root):
    frames = []

    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vidcap = cv2.VideoCapture(frame_root)

        while vidcap.isOpened():
            ret, frame = vidcap.read()

            if not ret:
                break

            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        vidcap.release()
    else:
        video_name = frame_root
        fr_lst = sorted(os.listdir(frame_root))

        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)

        fps = None

    size = frames[0].size
    return frames, fps, size, video_name

def resize_frames(frames, size=None):
    if size is not None:
        out_size = size
    else:
        out_size = frames[0].size

    process_size = (WIDTH, HEIGHT)

    frames = [f.resize(process_size) for f in frames]

    return frames, process_size, out_size

def read_mask(mpath, size):
    masks = []
    mnames = sorted(os.listdir(mpath))

    for m in mnames:
        m = Image.open(os.path.join(mpath, m))
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m * 255))

    return masks

def get_ref_index(f, neighbor_ids, length, num_ref, ref_length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if not i in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)

    return ref_index


# ======================
# Main functions
# ======================

def recognize_from_frames(net):
    neighbor_stride = args.neighbor_stride

    # prepare input frames
    frames, fps, size, video_name = read_frame_from_videos(args.video)

    if not args.width == -1 and not args.height == -1:
        size = (args.width, args.height)

    if not args.resize_ratio == 1.0:
        size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))

    frames, size, out_size = resize_frames(frames, size)

    video_length = len(frames)
    imgs = np.expand_dims(to_tensors()(frames), 0) * 2 - 1
    frames = [np.array(f).astype(np.uint8) for f in frames]

    fps = args.save_fps if fps is None else fps

    # prepare mask
    masks = read_mask(args.mask, size)
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    masks = np.expand_dims(to_tensors()(masks), 0)

    comp_frames = [None] * video_length
    logger.info(f'Loading videos and masks from: {args.video}')
    logger.info(f'Processing: {video_name} [{video_length} frames]...')

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time = 0
        total_timestamps = 0

    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, args.num_ref, args.step)
        logger.debug(f'{f} {len(neighbor_ids)} {len(ref_ids)}')
        selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]

        masked_imgs = selected_imgs * (1-selected_masks)

        start = int(round(time.time() * 1000))
        if args.onnx:
            pred_img = net.run(None, {net.get_inputs()[0].name: masked_imgs})[0]
        else:
            pred_img = net.run(masked_imgs)[0]
        end = int(round(time.time() * 1000))

        if args.benchmark:
            logger.info(f'\tailia processing time {end - start} ms for {pred_img.shape[0]} timestamps')
            total_time = total_time + (end - start)
            total_timestamps += pred_img.shape[0]

        pred_img = (pred_img + 1) / 2
        pred_img = pred_img.transpose((0, 2, 3, 1)) * 255

        for i in range(len(neighbor_ids)):
            idx = neighbor_ids[i]
            img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[idx] + frames[idx] * (1-binary_masks[idx])

            if comp_frames[idx] is None:
                comp_frames[idx] = img
            else:
                comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5

    if args.benchmark:
        logger.info(f'\taverage time {total_time / video_length} ms per frame (for a total of {total_timestamps} timestamps)')

    # save each frame
    if args.save_frames:
        frames_savepath = get_savepath(os.path.dirname(args.savepath), video_name, ext='', post_fix='_out_frames')

        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation = cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frame_savepath = get_savepath(frames_savepath, str(idx).zfill(5), ext='.jpg', post_fix='')
            cv2.imwrite(frame_savepath, f)

    savepath = get_savepath(os.path.dirname(args.savepath), video_name, ext='.mp4', post_fix='_out')
    logger.info(f'saved at : {savepath}')
    writer = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)

    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(np.uint8) * binary_masks[f] + frames[f] * (1-binary_masks[f])

        if comp.shape != out_size:
            comp = cv2.resize(comp, out_size, interpolation=cv2.INTER_LINEAR)

        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))

    writer.release()
    logger.info('Script finished successfully.')


def main():
    if args.video is None:
        logger.info(f'No input video or frames folder detected, running inference on {VIDEO_PATH}')
        args.video = VIDEO_PATH

    # model files check and download
    check_and_download_models(WEIGHT_FUSEFORMER_PATH, MODEL_FUSEFORMER_PATH, REMOTE_PATH)

    # net initialize
    if args.onnx:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_FUSEFORMER_PATH, providers=providers)
    else:
        net = ailia.Net(MODEL_FUSEFORMER_PATH, WEIGHT_FUSEFORMER_PATH, env_id=args.env_id)

    recognize_from_frames(net)


if __name__ == '__main__':
    main()
