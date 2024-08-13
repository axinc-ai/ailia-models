import numpy as np
import cv2
import sys
from typing import Any
from tqdm import tqdm

import onnxruntime
from ani_portrait_utils import get_model_file_names
from lmk_extractor import LMKExtractor
from facemesh_v2_utils import matrix_to_euler_and_translation, smooth_pose_seq, crop_face
from scipy.interpolate import interp1d

sys.path.append("../../util")
from detector_utils import load_image
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from scheduling_ddim import DDIMScheduler
import ailia

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/AniPortrait/"
FACEMESH_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/facemesh_v2"
MODES = ["Audio2Video", "Video2Video"]
INPUT_IMAGE = "lyl.png"
OUTPUT_IMAGE = ""
REF_IMAGE_SAMPLE = "lyl.png"
AUDIO_SAMPLE = "lyl.wav"
HEAD_POSE_SAMPLE = "pose_ref_video.mp4"

parser = get_base_parser("gpt2 text generation", INPUT_IMAGE, OUTPUT_IMAGE)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime",
)
parser.add_argument("-r", "--reference_image", type=str, default=REF_IMAGE_SAMPLE)
parser.add_argument("-hp", "--head_pose_reference_video", type=str, default=None)
parser.add_argument("-a", "--audio", default=AUDIO_SAMPLE)
parser.add_argument("-v", "--source_video")
parser.add_argument("-s", "--steps", type=int, default=25)
parser.add_argument("-S", "--seed", type=int, default=42)
parser.add_argument("-vs", "--video_size", type=int, default=512)
parser.add_argument("-l", "--length", type=int, default=0)
parser.add_argument("-m", "--mode", choices=MODES)
# parser.add_argument("-p", "--prompt", help="prompt text", required=True, type=str)
args = update_parser(parser, check_input_type=False)


def get_head_pose(lmk_extractor):
    trans_mat = []
    cap = cv2.VideoCapture(args.head_pose_reference_video)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        # img = load_image(args.input[0])
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        _trans_mat, _ = lmk_extractor(img)
        trans_mat.append(_trans_mat)

    cap.release()
    trans_mat = np.array(trans_mat)
    
    # Compute delta pose
    trans_mat_inv_frame_0 = np.linalg.inv(trans_mat[0])
    pose_arr = np.zeros([trans_mat.shape[0], 6])

    for i in range(pose_arr.shape[0]):
        pose_mat = trans_mat_inv_frame_0 @ trans_mat[i]
        euler_angles, translation_vector = matrix_to_euler_and_translation(pose_mat)
        pose_arr[i, :3] = euler_angles
        pose_arr[i, 3:6] = translation_vector

    new_fps = 30
    old_time = np.linspace(0, total_frames / fps, total_frames)
    new_time = np.linspace(0, total_frames / fps, int(total_frames * new_fps / fps))

    pose_arr_interp = np.zseros((len(new_time), 6))
    for i in range(6):
        interp_func = interp1d(old_time, pose_arr[:, i])
        pose_arr_interp[:, i] = interp_func(new_time)

    pose_arr_smooth = smooth_pose_seq(pose_arr_interp)
    return pose_arr_smooth


def generate_from_image(nets: dict[str, Any]):
    if args.mode == "Audio2Video":
        lmk_extractor = LMKExtractor(
            nets["face_landmarks_detector"],
            nets["face_detector"],
            args.onnx,
        )
        ref_image = load_image(args.reference_image)
        ref_image = crop_face(ref_image, lmk_extractor)
        fps = 30
        cfg = 3.5

        _, lmks = lmk_extractor(ref_image)
    else:
        pass


if __name__ == "__main__":
    model_file_names = get_model_file_names()

    nets = {}

    for root_model, _model_file_names in tqdm(model_file_names.items()):
        for model_name, model_files in tqdm(_model_file_names.items(), leave=False):
            # check_and_download_models(
            #     model_files["weight"],
            #     model_files["model"],
            #     REMOTE_PATH if root_model == "aniportrait" else FACEMESH_REMOTE_PATH,
            # )

            if args.onnx:
                net = onnxruntime.InferenceSession(model_files["weight"])
            else:
                net = ailia.Net(model_files["model"], model_files["weight"], env_id=args.env_id)

            nets[model_name] = net


    generate_from_image(nets)