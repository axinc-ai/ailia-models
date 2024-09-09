import sys
import os
import time
from logging import getLogger

import numpy as np
import cv2

import ailia

import os
import numpy as np
import matplotlib.pyplot as plt
import ailia

from tqdm import tqdm

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/segment-anything-2/'

IMAGE_PATH = 'truck.jpg'
SAVE_IMAGE_PATH = 'output.png'

POINT1 = (500, 375)
POINT2 = (1125, 625)

TARGET_LENGTH = 1024

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Segment Anything 2', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-p', '--pos', action='append', type=int, metavar="X", nargs=2,
    help='Positive coordinate specified by x,y.'
)
parser.add_argument(
    '--neg', action='append', type=int, metavar="X", nargs=2,
    help='Negative coordinate specified by x,y.'
)
parser.add_argument(
    '--box', type=int, metavar="X", nargs=4,
    help='Box coordinate specified by x1,y1,x2,y2.'
)
parser.add_argument(
    '--idx', type=int, choices=(0, 1, 2, 3),
    help='Select mask index.'
)
parser.add_argument(
    '-m', '--model_type', default='hiera_l', choices=('hiera_l', 'hiera_b+', 'hiera_s', 'hiera_t'),
    help='Select model.'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--normal', action='store_true',
    help='Use normal version of onnx model. Normal version requires 6 dim matmul.'
)

args = update_parser(parser)

# ======================
# Utility
# ======================

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, savepath = None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        if i == 0:
            plt.savefig(savepath)

# ======================
# Logic
# ======================

from sam2_image_predictor import SAM2ImagePredictor
from sam2_video_predictor import SAM2VideoPredictor

# ======================
# Main
# ======================

def recognize_from_image(image_encoder, prompt_encoder, mask_decoder):
    pos_points = args.pos
    neg_points = args.neg
    box = args.box

    if pos_points is None:
        if neg_points is None and box is None:
            pos_points = [POINT1]
        else:
            pos_points = []
    if neg_points is None:
        neg_points = []

    input_point = []
    input_label = []
    if pos_points:
        input_point.append(np.array(pos_points))
        input_label.append(np.ones(len(pos_points)))
    if neg_points:
        input_point.append(np.array(neg_points))
        input_label.append(np.zeros(len(neg_points)))
    input_point = np.array(input_point)
    input_label = np.array(input_label)
    if box:
        box = np.array(box)

    image_predictor = SAM2ImagePredictor()

    for image_path in args.input:
        image = cv2.imread(image_path)
        orig_hw = [image.shape[0], image.shape[1]]

        features = image_predictor.set_image(image, image_encoder, args.onnx)

        masks, scores, logits = image_predictor.predict(
            orig_hw=orig_hw,
            features=features,
            point_coords=input_point,
            point_labels=input_label,
            box=box,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            onnx=args.onnx
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, box_coords=box, borders=True, savepath=savepath)


def _load_img_as_tensor(img_path, image_size):
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img = cv2.imread(img_path)
    video_height = img.shape[0]  # the original video size
    video_width = img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img / 255.0
    img = img - img_mean
    img = img / img_std
    img = np.transpose(img, (2, 0, 1))
    return img, video_height, video_width


def load_video_frames(
    video_path,
    image_size,
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    if isinstance(video_path, str) and os.path.isdir(video_path):
        jpg_folder = video_path
    else:
        raise NotImplementedError(
            "Only JPEG frames are supported at this moment. For video files, you may use "
            "ffmpeg (https://ffmpeg.org/) to extract frames into a folder of JPEG files, such as \n"
            "```\n"
            "ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'\n"
            "```\n"
            "where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks "
            "ffmpeg to start the JPEG file from 00000.jpg."
        )

    frame_names = [
        p
        for p in os.listdir(jpg_folder)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {jpg_folder}")
    img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]    
    images = np.zeros((num_frames, 3, image_size, image_size), dtype=np.float32)
    for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    return images, video_height, video_width

def recognize_from_video(image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp):
    video_path = "./bedroom_short"
    image_size = 1024

    images, video_height, video_width = load_video_frames(
        video_path=video_path,
        image_size=image_size
    )

    from PIL import Image
    predictor = SAM2VideoPredictor(args.onnx, args.normal)

    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state()
    for i in range(len(images)):
        predictor.append_image(
            inference_state,
            images[i],
            video_height,
            video_width,
            image_encoder)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
    # sending all clicks (and their labels) to `add_new_points_or_box`
    points = np.array([[210, 350], [250, 220]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        mlp=mlp
    )

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_path, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    #plt.show()
    plt.savefig(f'video.png')

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                    image_encoder = image_encoder,
                                                                                    prompt_encoder = prompt_encoder,
                                                                                    mask_decoder = mask_decoder,
                                                                                    memory_attention = memory_attention,
                                                                                    memory_encoder = memory_encoder,
                                                                                    mlp = mlp):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_path, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        #plt.show()
        plt.savefig(f'video{out_frame_idx+1}_.png')

def main():
    # fetch image encoder model
    WEIGHT_IMAGE_ENCODER_L_PATH = 'image_encoder_'+args.model_type+'.onnx'
    MODEL_IMAGE_ENCODER_L_PATH = 'image_encoder_'+args.model_type+'.onnx.prototxt'
    WEIGHT_PROMPT_ENCODER_L_PATH = 'prompt_encoder_'+args.model_type+'.onnx'
    MODEL_PROMPT_ENCODER_L_PATH = 'prompt_encoder_'+args.model_type+'.onnx.prototxt'
    WEIGHT_MASK_DECODER_L_PATH = 'mask_decoder_'+args.model_type+'.onnx'
    MODEL_MASK_DECODER_L_PATH = 'mask_decoder_'+args.model_type+'.onnx.prototxt'
    if args.normal:
        # 6dim matmul
        WEIGHT_MEMORY_ATTENTION_L_PATH = 'memory_attention_'+args.model_type+'.onnx'
        MODEL_MEMORY_ATTENTION_L_PATH = 'memory_attention_'+args.model_type+'.onnx.prototxt'
    else:
        # 4dim matmul with batch 1
        WEIGHT_MEMORY_ATTENTION_L_PATH = 'memory_attention_'+args.model_type+'.opt.onnx'
        MODEL_MEMORY_ATTENTION_L_PATH = 'memory_attention_'+args.model_type+'.opt.onnx.prototxt'
    WEIGHT_MEMORY_ENCODER_L_PATH = 'memory_encoder_'+args.model_type+'.onnx'
    MODEL_MEMORY_ENCODER_L_PATH = 'memory_encoder_'+args.model_type+'.onnx.prototxt'
    WEIGHT_MLP_L_PATH = 'mlp_'+args.model_type+'.onnx'
    MODEL_MLP_L_PATH = 'mlp_'+args.model_type+'.onnx.prototxt'

    # model files check and download
    check_and_download_models(WEIGHT_IMAGE_ENCODER_L_PATH, MODEL_IMAGE_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_PROMPT_ENCODER_L_PATH, MODEL_PROMPT_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MASK_DECODER_L_PATH, MODEL_MASK_DECODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MEMORY_ATTENTION_L_PATH, MODEL_MEMORY_ATTENTION_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MEMORY_ENCODER_L_PATH, MODEL_MEMORY_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MLP_L_PATH, MODEL_MLP_L_PATH, REMOTE_PATH)

    if args.onnx:
        import onnxruntime
        image_encoder = onnxruntime.InferenceSession(WEIGHT_IMAGE_ENCODER_L_PATH)
        prompt_encoder = onnxruntime.InferenceSession(WEIGHT_PROMPT_ENCODER_L_PATH)
        mask_decoder = onnxruntime.InferenceSession(WEIGHT_MASK_DECODER_L_PATH)
        memory_attention = onnxruntime.InferenceSession(WEIGHT_MEMORY_ATTENTION_L_PATH)
        memory_encoder = onnxruntime.InferenceSession(WEIGHT_MEMORY_ENCODER_L_PATH)
        mlp = onnxruntime.InferenceSession(WEIGHT_MLP_L_PATH)
    else:
        memory_mode = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True, reduce_interstage=False, reuse_interstage=True)
        image_encoder = ailia.Net(weight=WEIGHT_IMAGE_ENCODER_L_PATH, stream=MODEL_IMAGE_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        prompt_encoder = ailia.Net(weight=WEIGHT_PROMPT_ENCODER_L_PATH, stream=MODEL_PROMPT_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        mask_decoder = ailia.Net(weight=WEIGHT_MASK_DECODER_L_PATH, stream=MODEL_MASK_DECODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        memory_attention = ailia.Net(weight=WEIGHT_MEMORY_ATTENTION_L_PATH, stream=MODEL_MEMORY_ATTENTION_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        memory_encoder = ailia.Net(weight=WEIGHT_MEMORY_ENCODER_L_PATH, stream=MODEL_MEMORY_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        mlp = ailia.Net(weight=WEIGHT_MLP_L_PATH, stream=MODEL_MLP_L_PATH, memory_mode=memory_mode, env_id=args.env_id)

    if args.video is not None:
        recognize_from_video(image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp)
    else:
        recognize_from_image(image_encoder, prompt_encoder, mask_decoder)

    logger.info('Script finished successfully.')

if __name__ == '__main__':
    main()
