from argparse import ArgumentParser
import time
import os
import json
import urllib.request

import cv2
import numpy as np

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses

import ailia


# Fixed when exporting ONNX model
base_height = 360
base_width = 640


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


def main():
    parser = ArgumentParser(
        description='Lightweight 3D human pose estimation demo. '
        'Press esc to exit, "p" to (un)pause video or process next image.'
    )
    parser.add_argument(
        '--video',
        help='Optional. Path to video file or camera id.',
        type=str,
        default=''
    )
    parser.add_argument(
        '--images',
        help='Optional. Path to input image(s).',
        nargs='+',
        default=''
    )
    parser.add_argument(
        '--rotate3d',
        help='allowing 3D canvas rotation while on pause',
        action='store_true',
        default=False
    )
    args = parser.parse_args()

    if args.images == '' and args.video == '':
        raise ValueError('Either --images or --video has to be provided')
    
    weight_path = 'human-pose-estimation-3d.onnx'
    model_path = 'human-pose-estimation-3d.onnx.prototxt'

    rmt_ckpt = "https://storage.googleapis.com/ailia-models/" +\
        "lightweight-human-pose-estimation-3d/"

    if not os.path.exists(model_path):
        print('downloading model...')
        urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
    if not os.path.exists(weight_path):
        print('downloading weight...')
        urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id (0: cpu, 1: gpu): {env_id}')
    net = ailia.Net(model_path, weight_path, env_id=env_id)

    stride = 8
    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    cv2.namedWindow(canvas_3d_window_name)
    cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    file_path = 'extrinsics.json'
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    frame_provider = ImageReader(args.images)
    is_video = False
    if args.video != '':
        frame_provider = VideoReader(args.video)
        is_video = True
    fx = -1

    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    img_mean = np.array([128, 128, 128], dtype=np.float32)
    
    for frame_id, frame in enumerate(frame_provider):
        current_time = cv2.getTickCount()
        if frame is None:
            break
        
        # fixed when the model was exported
        frame = cv2.resize(frame, dsize=(base_width, base_height))

        input_scale = base_height / frame.shape[0]
        
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

        # normalize image between -0.5 and 0.5
        normalized_img = (frame.astype(np.float32) - img_mean) / 255.0
        normalized_img = np.expand_dims(
            # np.rollaxis(normalized_img, 2, 0),
            normalized_img.transpose(2, 0, 1),
            axis=0
        )
        if is_video:
            input_blobs = net.get_input_blob_list()
            net.set_input_blob_data(normalized_img, input_blobs[0])
            net.update()
            features, heatmaps, pafs = net.get_results()
        else:
            for i in range(5):
                start = int(round(time.time() * 1000))
                input_blobs = net.get_input_blob_list()
                net.set_input_blob_data(normalized_img, input_blobs[0])
                net.update()
                features, heatmaps, pafs = net.get_results()
                end = int(round(time.time() * 1000))
                print("ailia processing time {} ms".format(end - start))

        inference_result = (
            features[-1].squeeze(),
            heatmaps[-1].squeeze(),
            pafs[-1].squeeze()
        )
        
        poses_3d, poses_2d = parse_poses(
            inference_result,
            input_scale,
            stride,
            fx,
            is_video
        )
        edges = []
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            edges = (
                Plotter3d.SKELETON_EDGES +\
                19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))
            ).reshape((-1, 2))
        plotter.plot(canvas_3d, poses_3d, edges)
        if is_video:
            cv2.imshow(canvas_3d_window_name, canvas_3d)
        else:
            cv2.imwrite(canvas_3d_window_name + f'_{frame_id}.png', canvas_3d)

        draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount()-current_time)/cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        if is_video:
            cv2.imshow('ICV 3D Human Pose Estimation', frame)
        else:
            cv2.imwrite(f'ICV_3D_Human_Pose_Estimation_{frame_id}.png', frame)

        key = cv2.waitKey(delay)
        if key == esc_code:
            break
        if key == p_code:
            if delay == 1:
                delay = 0
            else:
                delay = 1
        # allow to rotate 3D canvas while on pause
        # TODO make argument to activate \this
        if delay == 0 and args.rotate3d:
            key = 0
            while (key != p_code
                   and key != esc_code
                   and key != space_code):
                plotter.plot(canvas_3d, poses_3d, edges)
                cv2.imshow(canvas_3d_window_name, canvas_3d)
                key = cv2.waitKey(33)
            if key == esc_code:
                break
            else:
                delay = 1
    

if __name__ == "__main__":
    main()
