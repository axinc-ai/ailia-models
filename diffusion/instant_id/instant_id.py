import sys
from typing import Any, List

import cv2
import numpy as np

from detection_utils import get_detection
from face import Face
from instant_id_utils import draw_kps, get_model_file_names, load_image, preprocess
from landmark_utils import (
    P2sRt,
    estimate_affine_matrix_3d23d,
    load_mean_lmk,
    matrix2angle,
    trans_points,
    transform_landmark,
)
from pipe import get_pipe
from recognition_utils import norm_crop

sys.path.append("../../util")
import onnxruntime
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

INPUT_IMAGE = "sample.jpg"
OUTPUT_IMAGE = "output.jpg"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/instant_id/"
DETECTION_INPUT_STD = 128.0
DETECTION_INPUT_MEAN = 127.5
RECOGNITION_INPUT_STD = 127.5
RECOGNITION_INPUT_MEAN = 127.5
LANDMARK_3D_INPUT_STD = 1.0
LANDMARK_3D_INPUT_MEAN = 0.0
LANDMARK_3D_LMK_DIM = 3
LANDMARK_3D_LMK_NUM = 68
LANDMARK_3D_REQUIRE_POSE = True
LANDMARK_2D_INPUT_STD = 1.0
LANDMARK_2D_INPUT_MEAN = 0.0
LANDMARK_2D_LMK_DIM = 2
LANDMARK_2D_LMK_NUM = 106
LANDMARK_2D_REQUIRE_POSE = False
LANDMARK_INPUT_SIZE = (192, 192)
GENDERAGE_INPUT_SIZE = (96, 96)
GENDERAGE_INPUT_STD = 1.0
GENDERAGE_INPUT_MEAN = 0.0
INPUT_SIZE = (320, 320)
ADAPTER_STRENGTH_RATIO = 0.8
IDENTITYNET_STRENGTH_RATIO = 0.8


parser = get_base_parser("gpt2 text generation", INPUT_IMAGE, OUTPUT_IMAGE)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime",
)
parser.add_argument("-p", "--prompt", help="prompt text", required=True, type=str)
args = update_parser(parser, check_input_type=False)

img = load_image(INPUT_IMAGE)


def execute_detection(img: np.ndarray, detection_session):
    if args.onnx:
        input_name = detection_session.get_inputs()[0].name
        output_names = [output.name for output in detection_session.get_outputs()]
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / DETECTION_INPUT_STD,
            input_size,
            (DETECTION_INPUT_MEAN, DETECTION_INPUT_MEAN, DETECTION_INPUT_MEAN),
            swapRB=True,
        )
        net_outs = detection_session.run(output_names, {input_name: blob})
        return net_outs
    else:
        pass


def execute_recognition(img: np.ndarray, faces: List[Face], recognition_session):
    for face in faces:
        aimg = norm_crop(img, landmark=face.kps, image_size=112)
        blob = cv2.dnn.blobFromImages(
            [aimg],
            1.0 / RECOGNITION_INPUT_STD,
            (112, 112),
            (RECOGNITION_INPUT_MEAN, RECOGNITION_INPUT_MEAN, RECOGNITION_INPUT_MEAN),
            swapRB=True,
        )
        if args.onnx:
            input_name = recognition_session.get_inputs()[0].name
            output_name = recognition_session.get_outputs()[0].name
            net_out = recognition_session.run([output_name], {input_name: blob})[0]
            face.embedding = net_out.flatten()
        else:
            pass


def execute_landmark(img: np.ndarray, faces: List[Face], landmark_session, taskname):
    require_pose = (
        LANDMARK_3D_REQUIRE_POSE if "3d" in taskname else LANDMARK_2D_REQUIRE_POSE
    )
    mean_lmk = load_mean_lmk("meanshape_68.pkl") if require_pose else None
    lmk_num = LANDMARK_3D_LMK_NUM if require_pose else LANDMARK_2D_LMK_NUM

    for face in faces:
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = 192 / (max(w, h) * 1.5)
        aimg, M = transform_landmark(
            img, center, LANDMARK_INPUT_SIZE[0], _scale, rotate
        )
        input_size = tuple(aimg.shape[0:2][::-1])
        if taskname == "landmark_2d_106":
            blob = cv2.dnn.blobFromImage(
                aimg,
                1.0 / LANDMARK_2D_INPUT_STD,
                input_size,
                (
                    LANDMARK_2D_INPUT_MEAN,
                    LANDMARK_2D_INPUT_MEAN,
                    LANDMARK_2D_INPUT_MEAN,
                ),
                swapRB=True,
            )
        else:
            blob = cv2.dnn.blobFromImage(
                aimg,
                1.0 / LANDMARK_3D_INPUT_STD,
                input_size,
                (
                    LANDMARK_3D_INPUT_MEAN,
                    LANDMARK_3D_INPUT_MEAN,
                    LANDMARK_3D_INPUT_MEAN,
                ),
                swapRB=True,
            )

        if args.onnx:
            input_name = landmark_session.get_inputs()[0].name
            output_name = landmark_session.get_outputs()[0].name
            net_out = landmark_session.run([output_name], {input_name: blob})[0][0]
            if net_out.shape[0] >= 3000:
                net_out = net_out.reshape((-1, 3))
            else:
                net_out = net_out.reshape((-1, 2))
            if lmk_num < net_out.shape[0]:
                net_out = net_out[lmk_num * -1 :, :]

            net_out[:, 0:2] += 1
            net_out[:, 0:2] *= LANDMARK_INPUT_SIZE[0] // 2
            if net_out.shape[1] == 3:
                net_out[:, 2] *= LANDMARK_INPUT_SIZE[0] // 2

            IM = cv2.invertAffineTransform(M)
            net_out = trans_points(net_out, IM)
            face[taskname] = net_out
            if require_pose:
                P = estimate_affine_matrix_3d23d(mean_lmk, net_out)
                s, R, t = P2sRt(P)
                rx, ry, rz = matrix2angle(R)
                pose = np.array([rx, ry, rz], dtype=np.float32)
                face["pose"] = pose  # pitch, yaw, roll

        else:
            pass


def execute_genderage(img: np.ndarray, faces: List[Face], genderage_session):
    for face in faces:
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = GENDERAGE_INPUT_SIZE[0] / (max(w, h) * 1.5)
        aimg, M = transform_landmark(
            img, center, GENDERAGE_INPUT_SIZE[0], _scale, rotate
        )
        input_size = tuple(aimg.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / GENDERAGE_INPUT_STD,
            input_size,
            (GENDERAGE_INPUT_MEAN, GENDERAGE_INPUT_MEAN, GENDERAGE_INPUT_MEAN),
            swapRB=True,
        )

        if args.onnx:
            input_name = genderage_session.get_inputs()[0].name
            output_name = genderage_session.get_outputs()[0].name
            net_out = genderage_session.run([output_name], {input_name: blob})[0][0]
            gender = np.argmax(net_out[:2])
            age = int(np.round(net_out[2] * 100))
            face["gender"] = gender
            face["age"] = age


def infer_from_image(nets: dict[str, Any]):
    input_image, input_image_pil = load_image(args.input[0])
    preprocessed_image, det_scale = preprocess(input_image, INPUT_SIZE)
    detection_sess_out = execute_detection(preprocessed_image, nets["detection"])
    net_outs = {
        "detection": detection_sess_out,
    }
    detection_out = get_detection(net_outs, True, det_scale)
    execute_recognition(input_image, detection_out, nets["recognition"])
    execute_landmark(
        input_image, detection_out, nets["landmark_2d_106"], taskname="landmark_2d_106"
    )
    execute_landmark(
        input_image, detection_out, nets["landmark_3d_68"], taskname="landmark_3d_68"
    )
    execute_genderage(input_image, detection_out, nets["genderage"])

    face_info = sorted(
        detection_out,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
    )[-1]
    face_emb = face_info["embedding"]
    face_kps = draw_kps(input_image_pil, face_info["kps"])

    pipe = get_pipe()
    pipe.set_ip_adapter_scale(ADAPTER_STRENGTH_RATIO)
    images = pipe(
        args.prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=IDENTITYNET_STRENGTH_RATIO,
        num_inference_steps=1,
        guidance_scale=0.0,
    ).images[0]
    images.save(args.savepath)


if __name__ == "__main__":
    model_file_names = get_model_file_names()

    nets = {}

    for model_name, model_files in model_file_names.items():
        # check_and_download_models(
        #     model_files["weight"],
        #     model_files["model"],
        #     REMOTE_PATH,
        # )

        if args.onnx:
            net = onnxruntime.InferenceSession(model_files["weight"])
        else:
            net = None

        nets[model_name] = net

    infer_from_image(nets)
