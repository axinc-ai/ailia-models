import os, sys, time
from time import strftime
import shutil
import numpy as np
import random
from argparse import ArgumentParser

import ailia

# import original modules
sys.path.append('../../util')
import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

from preprocess import CropAndExtract
from audio2coeff import Audio2Coeff
from animation import AnimateFromCoeff
from batch_generation import get_data, get_facerender_data

# logger
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.png'
INPUT_AUDIO_PATH = "input.wav"
SAVE_IMAGE_PATH = 'output.mp4'

LM3D_PATH = "preprocess/similarity_Lm3D_all.mat"

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser("sadtalker", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument("-a", "--audio", default=INPUT_AUDIO_PATH, help="Path to input audio")
parser.add_argument("--result_dir", default='./results', help="path to output")
parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
parser.add_argument("--expression_scale", type=float, default=1.,  help="the value of expression intensity")
parser.add_argument("--batch_size", type=int, default=2, help="the batch size of facerender")
parser.add_argument("--size", type=int, default=256, help="the image size of the facerender")
parser.add_argument('--enhancer', type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]") #TODO
parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion")
parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" )
parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" )
args = update_parser(parser)

# ======================
# Parameters 2
# ======================
WEIGHT_FACE3D_RECON_PATH = "face3d_recon.onnx"
# MODEL_FACE3D_RECON_PATH = "face3d_recon.onnx.prototxt"
MODEL_FACE3D_RECON_PATH = None
WEIGHT_FACE_ALIGN_PATH = "face_align.onnx"
# MODEL_FACE_ALIGN_PATH = "face_align.onnx.prototxt"
MODEL_FACE_ALIGN_PATH = None
WEIGHT_AUDIO2EXP_PATH = "audio2exp.onnx"
# MODEL_AUDIO2EXP_PATH = "audio2exp.onnx.prototxt"
MODEL_AUDIO2EXP_PATH = None
WEIGHT_AUDIO2POSE_PATH = "audio2pose.onnx"
# MODEL_AUDIO2POSE_PATH = "audio2pose.onnx.prototxt"
MODEL_AUDIO2POSE_PATH = None
WEIGHT_ANIMATION_GENERATOR_PATH = "animation_generator.onnx"
# MODEL_ANIMATION_GENERATOR_PATH = "animation_generator.onnx.prototxt"
MODEL_ANIMATION_GENERATOR_PATH = None
WEIGHT_KP_DETECTOR_PATH = "kp_detector.onnx"
# MODEL_KP_DETECTOR_PATH = "kp_detector.onnx.prototxt"
MODEL_KP_DETECTOR_PATH = None
WEIGHT_MAPPING_NET = "mappingnet_full.onnx" if "full" in args.preprocess else "mappingnet_not_full.onnx"
# MODEL_MAPPING_NET = WEIGHT_MAPPING_NET + "prototxt"
MODEL_MAPPING_NET = None
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sadtalker/"

WEIGHT_FACE_DET_PATH = "retinaface_resnet50.onnx"
# MODEL_FACE_DET_PATH = "retinaface_resnet50.onnx.prototxt"
MODEL_FACE_DET_PATH = None
REMOTE_FACE_DET_PATH = "https://storage.googleapis.com/ailia-models/retinaface/"

WEIGHT_GFPGAN_PATH = "GFPGANv1.4.onnx"
# MODEL_GFPGAN_PATH = "GFPGANv1.4.onnx.prototxt"
MODEL_GFPGAN_PATH = None
REMOTE_GFPGAN_PATH = "https://storage.googleapis.com/ailia-models/gfpgan/"

# ======================
# Utils
# ======================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# ======================
# Main functions
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_FACE3D_RECON_PATH, MODEL_FACE3D_RECON_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_FACE_ALIGN_PATH, MODEL_FACE_ALIGN_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_AUDIO2EXP_PATH, MODEL_AUDIO2EXP_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_AUDIO2POSE_PATH, MODEL_AUDIO2POSE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ANIMATION_GENERATOR_PATH, MODEL_ANIMATION_GENERATOR_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_KP_DETECTOR_PATH, MODEL_KP_DETECTOR_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MAPPING_NET, MODEL_MAPPING_NET, REMOTE_PATH)

    check_and_download_models(WEIGHT_FACE_DET_PATH, MODEL_FACE_DET_PATH, REMOTE_FACE_DET_PATH)
    check_and_download_models(WEIGHT_GFPGAN_PATH, MODEL_GFPGAN_PATH, REMOTE_GFPGAN_PATH)

    # model initialize
    face3d_recon_net = ailia.Net(MODEL_FACE3D_RECON_PATH, WEIGHT_FACE3D_RECON_PATH, env_id=args.env_id)
    face_align_net = ailia.Net(MODEL_FACE_ALIGN_PATH, WEIGHT_FACE_ALIGN_PATH, env_id=args.env_id)
    audio2exp_net = ailia.Net(MODEL_AUDIO2EXP_PATH, WEIGHT_AUDIO2EXP_PATH, env_id=args.env_id)
    audio2pose_net = ailia.Net(MODEL_AUDIO2POSE_PATH, WEIGHT_AUDIO2POSE_PATH, env_id=args.env_id)
    generator_net = ailia.Net(MODEL_ANIMATION_GENERATOR_PATH, WEIGHT_ANIMATION_GENERATOR_PATH, env_id=args.env_id)
    kp_detector_net = ailia.Net(MODEL_KP_DETECTOR_PATH, WEIGHT_KP_DETECTOR_PATH, env_id=args.env_id)
    mapping_net = ailia.Net(MODEL_MAPPING_NET, WEIGHT_MAPPING_NET, env_id=args.env_id)

    retinaface_net = ailia.Net(MODEL_FACE_DET_PATH, WEIGHT_FACE_DET_PATH)
    gfpgan_net = ailia.Net(MODEL_GFPGAN_PATH, WEIGHT_GFPGAN_PATH)

    set_seed(42)

    pic_path = args.input[0]
    audio_path = args.audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    # batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    current_root_path = os.path.split(sys.argv[0])[0]

    #init model
    preprocess_model = CropAndExtract(face3d_recon_net, face_align_net, retinaface_net, 
                                      os.path.join(current_root_path, LM3D_PATH))
    audio_to_coeff = Audio2Coeff(audio2exp_net, audio2pose_net)
    animate_from_coeff = AnimateFromCoeff(generator_net, kp_detector_net, mapping_net)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,\
                                                                             source_image_flag=True, pic_size=args.size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                args.batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)
    
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info,
                                enhancer=args.enhancer, background_enhancer=None, preprocess=args.preprocess, img_size=args.size,
                                retinaface_net=retinaface_net, gfpgan_net=gfpgan_net)
    
    shutil.move(result, save_dir+'.mp4')
    print('The generated video is named:', save_dir+'.mp4')

    if not args.verbose:
        shutil.rmtree(save_dir)

    
if __name__ == '__main__':
    main()