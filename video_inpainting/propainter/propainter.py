import os
import sys
import time
from tqdm import tqdm

import cv2
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from propainter_utils import read_frame_from_videos, read_mask, resize_frames, extrapolation, to_tensors, get_ref_index
from propainter_nets import RecurrentFlowCompleteNet, InpaintGenerator

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_RAFT_PATH = 'raft.onnx'
MODEL_RAFT_PATH = 'raft.onnx.prototxt'

WEIGHT_COMPLETE_FLOW_PATH = 'complete_flow.onnx'
MODEL_COMPLETE_FLOW_PATH = 'complete_flow.onnx.prototxt'

WEIGHT_IMAGE_PROPAGATION_PATH = 'image_propagation.onnx'
MODEL_IMAGE_PROPAGATION_PATH = 'image_propagation.onnx.prototxt'

WEIGHT_PROPAINTER_PATH = 'propainter.onnx'
MODEL_PROPAINTER_PATH = 'propainter.onnx.prototxt'


REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/propainter/'

VIDEO_PATH = 'bmx_trees.mp4'
MASK_PATH = 'bmx_trees_mask'
SAVE_VIDEO_PATH = 'bmx_trees_out.mp4'


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser('ProPainter', None, SAVE_VIDEO_PATH)
parser.add_argument(
    '-m', '--mask',  type=str, default=MASK_PATH,
    help='Path of the mask(s) or mask folder.'
)
parser.add_argument(
    '--mode', default='video_inpainting', choices=('video_inpainting', 'video_outpainting'),
    help='Modes: video_inpainting / video_outpainting.'
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
    '--save_fps', type=int, default=24,
    help='Frame per second. Default: 24'
)
parser.add_argument(
    '--mask_dilation', type=int, default=4,
    help='Mask dilation for video and flow masking.'
)
parser.add_argument(
    '--scale_h', type=float, default=1.0,
    help='Outpainting scale of height for video_outpainting mode.'
)
parser.add_argument(
    '--scale_w', type=float, default=1.2,
    help='Outpainting scale of width for video_outpainting mode.'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='If set, runs the inference using onnx runtime.'
)
parser.add_argument(
    '--subvideo_length', type=int, default=80,
    help='Length of sub-video for long video inference.'
)
parser.add_argument(
    '--ref_stride', type=int, default=10,
    help='Stride of global reference frames.'
)
parser.add_argument(
    '--neighbor_length', type=int, default=10,
    help='Length of local neighboring frames.'
)
parser.add_argument(
    '--save_frames', action='store_true',
    help='Save output frames.'
)

args = update_parser(parser)


# ======================
# Helper functions
# ======================

def save_video(savepath, frames, video_name, fps, out_size):
    savepath = os.path.join(os.path.dirname(savepath), video_name)
    logger.info(f'saved at : {savepath}')
    writer = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)

    for f in frames:
        writer.write(cv2.cvtColor(np.array(f).astype(np.uint8), cv2.COLOR_BGR2RGB))

    writer.release()


# ======================
# Main functions
# ======================

def recognize_from_frames(nets):
    # prepare input frames
    frames, fps, size, video_name = read_frame_from_videos(args.video)

    if not args.width == -1 and not args.height == -1:
        size = (args.width, args.height)

    if not args.resize_ratio == 1.0:
        size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))

    frames, size, out_size = resize_frames(frames, size)

    fps = args.save_fps if fps is None else fps

    # prepare mask
    if args.mode == 'video_inpainting':
        frames_len = len(frames)
        flow_masks, masks_dilated = read_mask(args.mask, frames_len, size,
                                              flow_mask_dilates=args.mask_dilation,
                                              mask_dilates=args.mask_dilation)
        w, h = size
    elif args.mode == 'video_outpainting':
        assert args.scale_h is not None and args.scale_w is not None, 'Please provide an outpainting scale (s_h, s_w).'
        frames, flow_masks, masks_dilated, size = extrapolation(frames, (args.scale_h, args.scale_w))
        w, h = size
    else:
        raise NotImplementedError

    # preprocessing
    frames_inp = [np.array(f).astype(np.uint8) for f in frames]
    frames = np.expand_dims(to_tensors()(frames), 0) * 2 - 1
    flow_masks = np.expand_dims(to_tensors()(flow_masks), 0)
    masks_dilated = np.expand_dims(to_tensors()(masks_dilated), 0)

    video_length = frames.shape[1]
    logger.info(f'Processing: {video_name} [{video_length} frames]...')

    # inference
    logger.info('Start inference...')

    # ---- compute flow ----
    logger.debug(f'Computing flow...')
    if frames.shape[-1] <= 640:
        short_clip_len = 12
    elif frames.shape[-1] <= 720:
        short_clip_len = 8
    elif frames.shape[-1] <= 1280:
        short_clip_len = 4
    else:
        short_clip_len = 2

    if video_length > short_clip_len:
        gt_flows_f_list, gt_flows_b_list = [], []
        for f in tqdm(range(0, video_length, short_clip_len)):
            end_f = min(video_length, f + short_clip_len)
            if f == 0:
                if args.onnx:
                    flows_f, flows_b = nets['raft'].run(None, {nets['raft'].get_inputs()[0].name: frames[:,f:end_f]})
                else:
                    flows_f, flows_b = nets['raft'].run(frames[:,f:end_f])
            else:
                if args.onnx:
                    flows_f, flows_b = nets['raft'].run(None, {nets['raft'].get_inputs()[0].name: frames[:,f-1:end_f]})
                else:
                    flows_f, flows_b = nets['raft'].run(frames[:,f-1:end_f])

            gt_flows_f_list.append(flows_f)
            gt_flows_b_list.append(flows_b)

        gt_flows_f = np.concatenate(gt_flows_f_list, axis=1)
        gt_flows_b = np.concatenate(gt_flows_b_list, axis=1)
        gt_flows_bi = (gt_flows_f, gt_flows_b)
    else:
        if args.onnx:
            gt_flows_bi = nets['raft'].run(None, {nets['raft'].get_inputs()[0].name: frames})
        else:
            gt_flows_bi = nets['raft'].run(frames)

    logger.debug(f'Flow computed.')

    # ---- complete flow ----
    logger.debug(f'Completing flow...')
    fix_flow_complete = RecurrentFlowCompleteNet(nets['complete_flow'], args.onnx)
    flow_length = gt_flows_bi[0].shape[1]
    if flow_length > args.subvideo_length:
        pred_flows_f, pred_flows_b = [], []
        pad_len = 5
        for f in range(0, flow_length, args.subvideo_length):
            s_f = max(0, f - pad_len)
            e_f = min(flow_length, f + args.subvideo_length + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
            pred_flows_bi_sub = fix_flow_complete.forward_bidirect_flow(
                (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                flow_masks[:, s_f:e_f+1])
            pred_flows_bi_sub = fix_flow_complete.combine_flow(
                (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                pred_flows_bi_sub,
                flow_masks[:, s_f:e_f+1])

            pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
            pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])

        pred_flows_f = np.concatenate(pred_flows_f, axis=1)
        pred_flows_b = np.concatenate(pred_flows_b, axis=1)
        pred_flows_bi = (pred_flows_f, pred_flows_b)
    else:
        pred_flows_bi = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
        pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)

    logger.debug(f'Flow completed.')

    # ---- image propagation ----
    logger.debug(f'Image propagation...')
    model = InpaintGenerator(nets['image_propagation'], nets['propainter'], args.onnx)
    masked_frames = frames * (1 - masks_dilated)
    subvideo_length_img_prop = min(100, args.subvideo_length) # ensure a minimum of 100 frames for image propagation
    if video_length > subvideo_length_img_prop:
        updated_frames, updated_masks = [], []
        pad_len = 10
        for f in range(0, video_length, subvideo_length_img_prop):
            s_f = max(0, f - pad_len)
            e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

            b, t, _, _, _ = masks_dilated[:, s_f:e_f].shape
            pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
            prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_frames[:, s_f:e_f],
                                                                    pred_flows_bi_sub,
                                                                    masks_dilated[:, s_f:e_f])
            updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                np.reshape(prop_imgs_sub, (b, t, 3, h, w)) * masks_dilated[:, s_f:e_f]
            updated_masks_sub = np.reshape(updated_local_masks_sub, (b, t, 1, h, w))

            updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
            updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])

        updated_frames = np.concatenate(updated_frames, axis=1)
        updated_masks = np.concatenate(updated_masks, axis=1)
    else:
        b, t, _, _, _ = masks_dilated.shape
        prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated)

        updated_frames = frames * (1 - masks_dilated) + np.reshape(prop_imgs, (b, t, 3, h, w)) * masks_dilated
        updated_masks = np.reshape(updated_local_masks, (b, t, 1, h, w))

    logger.debug(f'Image propagation done.')

    ori_frames = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = args.neighbor_length // 2
    if video_length > args.subvideo_length:
        ref_num = args.subvideo_length // args.ref_stride
    else:
        ref_num = -1

    # ---- feature propagation + transformer ----
    logger.debug(f'Feature propagation...')
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

        # 1.0 indicates mask
        l_t = len(neighbor_ids)

        if args.onnx:
            pred_img = nets['propainter'].run(None, {nets['propainter'].get_inputs()[0].name: selected_imgs,
                                           nets['propainter'].get_inputs()[1].name: selected_pred_flows_bi[0],
                                           nets['propainter'].get_inputs()[2].name: selected_pred_flows_bi[1],
                                           nets['propainter'].get_inputs()[3].name: selected_masks,
                                           nets['propainter'].get_inputs()[4].name: selected_update_masks,
                                           nets['propainter'].get_inputs()[5].name: np.array([l_t]).astype(np.int64)})[0]
        else:
            pred_img = nets['propainter'].run(selected_imgs,
                                              selected_pred_flows_bi[0],
                                              selected_pred_flows_bi[1],
                                              selected_masks,
                                              selected_update_masks,
                                              np.array([l_t]).astype(np.int64))[0]

        pred_img = np.reshape(pred_img, (-1, 3, h, w))

        pred_img = (pred_img + 1) / 2
        pred_img = pred_img.transpose((0, 2, 3, 1)) * 255
        binary_masks = masks_dilated[0, neighbor_ids, :, :, :].transpose((0, 2, 3, 1)).astype(np.uint8)

        for i in range(len(neighbor_ids)):
            idx = neighbor_ids[i]
            img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] + ori_frames[idx] * (1 - binary_masks[i])

            if comp_frames[idx] is None:
                comp_frames[idx] = img
            else:
                comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5

            comp_frames[idx] = comp_frames[idx].astype(np.uint8)

        logger.debug(f'Feature propagation done.')

    # save each frame
    if args.save_frames:
        frames_savepath = get_savepath(os.path.dirname(args.savepath), video_name, ext='', post_fix='_out_frames')

        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation = cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frame_savepath = get_savepath(frames_savepath, str(idx).zfill(5), ext='.jpg', post_fix='')
            cv2.imwrite(frame_savepath, f)

    # save videos frame
    masked_frame_for_save = [cv2.resize(f, out_size) for f in masked_frame_for_save]
    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]

    save_video(args.savepath, masked_frame_for_save, 'masked_in.mp4', fps, out_size)
    save_video(args.savepath, comp_frames, 'inpaint_out.mp4', fps, out_size)

    logger.info('Script finished successfully.')


def main():
    if args.video is None:
        logger.info(f'No input video or frames folder detected, running inference on {VIDEO_PATH}')
        args.video = VIDEO_PATH

    # model files check and download
    # check_and_download_models(WEIGHT_RAFT_PATH, MODEL_RAFT_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_COMPLETE_FLOW_PATH, MODEL_COMPLETE_FLOW_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_IMAGE_PROPAGATION_PATH, MODEL_IMAGE_PROPAGATION_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_PROPAINTER_PATH, MODEL_PROPAINTER_PATH, REMOTE_PATH)

    # net initialize
    if args.onnx:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        raft_model = onnxruntime.InferenceSession(WEIGHT_RAFT_PATH, providers=providers)
        complete_flow_model = onnxruntime.InferenceSession(WEIGHT_COMPLETE_FLOW_PATH, providers=providers)
        image_propagation_model = onnxruntime.InferenceSession(WEIGHT_IMAGE_PROPAGATION_PATH, providers=providers)
        propainter_model = onnxruntime.InferenceSession(WEIGHT_PROPAINTER_PATH, providers=providers)
    else:
        raft_model = ailia.Net(None, WEIGHT_RAFT_PATH, env_id=args.env_id)
        complete_flow_model = ailia.Net(None, WEIGHT_COMPLETE_FLOW_PATH, env_id=args.env_id)
        image_propagation_model = ailia.Net(None, WEIGHT_IMAGE_PROPAGATION_PATH, env_id=args.env_id)
        propainter_model = ailia.Net(None, WEIGHT_PROPAINTER_PATH, env_id=args.env_id)

    nets = {
        'raft': raft_model,
        'complete_flow': complete_flow_model,
        'image_propagation': image_propagation_model,
        'propainter': propainter_model
    }

    recognize_from_frames(nets)


if __name__ == '__main__':
    main()
