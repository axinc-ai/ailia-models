import cv2
import sys
import numpy as np
import pandas as pd
import os
import time
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import urllib.request
import math
import argparse

from mpl_toolkits.mplot3d import Axes3D

import ailia

sys.path.append('../util')
from webcamera_utils import adjust_frame_size  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import check_file_existance  # noqa: E402



# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

ALGORITHM = ailia.POSE_ALGORITHM_LW_HUMAN_POSE


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Fast and accurate human pose 2D-estimation.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=IMAGE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='The input video path. ' +
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
args = parser.parse_args()


# ======================
# Parameters 2
# ======================
MODEL_NAME = 'lightweight-human-pose-estimation'
if args.normal:
    WEIGHT_PATH = f'{MODEL_NAME}.onnx'
    MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'
else:
    WEIGHT_PATH = f'{MODEL_NAME}.opt.onnx'
    MODEL_PATH = f'{MODEL_NAME}.opt.onnx.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'

BASELINE_MODEL_NAME = '3d-pose-baseline'
BASELINE_WEIGHT_PATH = f'{BASELINE_MODEL_NAME}.onnx'
BASELINE_MODEL_PATH = f'{BASELINE_MODEL_NAME}.onnx.prototxt'
BASELINE_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{BASELINE_MODEL_NAME}/'



# ======================
# 3d-pose Utils
# ======================

with h5py.File('3d-pose-baseline-mean.h5', 'r') as f:
  data_mean_2d = np.array(f['data_mean_2d'])
  data_std_2d = np.array(f['data_std_2d'])
  data_mean_3d = np.array(f['data_mean_3d'])
  data_std_3d = np.array(f['data_std_3d'])

H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip' #ignore when 3d
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose' #ignore when 2d
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

h36m_2d_mean = [0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27]
h36m_3d_mean = [1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]

OPENPOSE_Nose = 0
OPENPOSE_Neck = 1
OPENPOSE_RightShoulder = 2
OPENPOSE_RightElbow = 3
OPENPOSE_RightWrist = 4
OPENPOSE_LeftShoulder = 5
OPENPOSE_LeftElbow = 6
OPENPOSE_LeftWrist = 7
OPENPOSE_RightHip = 8
OPENPOSE_RightKnee = 9
OPENPOSE_RightAnkle = 10
OPENPOSE_LeftHip = 11
OPENPOSE_LeftKnee = 12
OPENPOSE_LAnkle = 13
OPENPOSE_RightEye = 14
OPENPOSE_LeftEye = 15
OPENPOSE_RightEar = 16
OPENPOSE_LeftEar = 17
OPENPOSE_Background = 18

openpose_to_3dposebaseline=[-1,8,9,10,11,12,13,-1,1,0,5,6,7,2,3,4]

fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(18, -70)  

def search_name(name,IS_3D):
    j=0
    for i in range(32):
        if(IS_3D):
            if(H36M_NAMES[i]=="Hip"):
                continue
        else:
            if(H36M_NAMES[i]=="Neck/Nose"):
                continue
        if(H36M_NAMES[i]==""):
            continue
        if(H36M_NAMES[i]==name):
            return j
        j=j+1
    return -1

def draw_connect(from_id,to_id,color,X,Y,Z,IS_3D):
    from_id=search_name(from_id,IS_3D)
    to_id=search_name(to_id,IS_3D)
    if(from_id==-1 or to_id==-1):
        return
    x = [X[from_id], X[to_id]]
    y = [Y[from_id], Y[to_id]]
    z = [Z[from_id], Z[to_id]]

    ax.plot(x, z, y, "o-", color=color, ms=4, mew=0.5)

def plot(outputs,inputs):
    plt.cla()

    cnt=0

    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.set_zlabel('Y axis')
    ax.set_zlim([600, -600])

    #global cnt,X,Y,Z,IS_3D
    #k=cnt

    for mode in range(2):
        X=[]
        Y=[]
        Z=[]

        if(mode==0):
            IS_3D=True
        else:
            IS_3D=False

        for i in range(16):
            if IS_3D:
                X.append(outputs[i*3+0])
                Y.append(outputs[i*3+1])
                Z.append(outputs[i*3+2])
            else:
                j=h36m_2d_mean[i]
                X.append(inputs[i*2+0]*data_std_2d[j*2+0]+data_mean_2d[j*2+0])
                Y.append(inputs[i*2+1]*data_std_2d[j*2+1]+data_mean_2d[j*2+1])
                Z.append(0)

        if(IS_3D):
            draw_connect("Head","Thorax","#0000aa",X,Y,Z,IS_3D)
            draw_connect("Thorax",'RShoulder',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('RShoulder','RElbow',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('RElbow','RWrist',"#00ff00",X,Y,Z,IS_3D)
            draw_connect("Thorax",'LShoulder',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('LShoulder','LElbow',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('LElbow','LWrist',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('Thorax','Spine',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('Spine','LHip',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('Spine','RHip',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('RHip','RKnee',"#ff0000",X,Y,Z,IS_3D)
            draw_connect('RKnee','RFoot',"#ff0000",X,Y,Z,IS_3D)
            draw_connect('LHip','LKnee',"#ff0000",X,Y,Z,IS_3D)
            draw_connect('LKnee','LFoot',"#ff0000",X,Y,Z,IS_3D)
        else:
            draw_connect("Head","Thorax","#0000ff",X,Y,Z,IS_3D)
            draw_connect("Thorax",'RShoulder',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('RShoulder','RElbow',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('RElbow','RWrist',"#00ff00",X,Y,Z,IS_3D)
            draw_connect("Thorax",'LShoulder',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('LShoulder','LElbow',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('LElbow','LWrist',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('Thorax','Spine',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('Spine','Hip',"#00ff00",X,Y,Z,IS_3D)
            draw_connect('Hip','LHip',"#ff0000",X,Y,Z,IS_3D)
            draw_connect('Hip','RHip',"#ff0000",X,Y,Z,IS_3D)
            draw_connect('RHip','RKnee',"#ff0000",X,Y,Z,IS_3D)
            draw_connect('RKnee','RFoot',"#ff0000",X,Y,Z,IS_3D)
            draw_connect('LHip','LKnee',"#ff0000",X,Y,Z,IS_3D)
            draw_connect('LKnee','LFoot',"#ff0000",X,Y,Z,IS_3D)

def display_3d_pose(points,baseline):
    points_queue=points

    inputs = np.zeros(32)

    for i in range(16):
        if openpose_to_3dposebaseline[i]==-1:
            continue
        inputs[i*2+0]=points[openpose_to_3dposebaseline[i]*2+0]
        inputs[i*2+1]=points[openpose_to_3dposebaseline[i]*2+1]

    inputs[0*2+0] = (points[11*2+0]+points[8*2+0])/2
    inputs[0*2+1] = (points[11*2+1]+points[8*2+1])/2
    inputs[7*2+0] = (points[5*2+0]+points[2*2+0])/2
    inputs[7*2+1] = (points[5*2+1]+points[2*2+1])/2

    spine_x = inputs[24]
    spine_y = inputs[25]

    for i in range(16):
        j=h36m_2d_mean[i]
        target_width=600
        target_height=600
        inputs[i*2+0]=(inputs[i*2+0]*target_width-data_mean_2d[j*2+0])/data_std_2d[j*2+0]
        inputs[i*2+1]=(inputs[i*2+1]*target_height-data_mean_2d[j*2+1])/data_std_2d[j*2+1]

    reshape_input = np.reshape(np.array(inputs),(1,32))

    outputs = baseline.predict(reshape_input)[0]

    for i in range(16):
        j=h36m_3d_mean[i]
        outputs[i*3+0]=outputs[i*3+0]*data_std_3d[j*3+0]+data_mean_3d[j*3+0]
        outputs[i*3+1]=outputs[i*3+1]*data_std_3d[j*3+1]+data_mean_3d[j*3+1]
        outputs[i*3+2]=outputs[i*3+2]*data_std_3d[j*3+2]+data_mean_3d[j*3+2]

    for i in range(16):
        dx = outputs[i*3+0] - data_mean_3d[0*3+0]
        dy = outputs[i*3+1] - data_mean_3d[0*3+1]
        dz = outputs[i*3+2] - data_mean_3d[0*3+2]

        theta = math.radians(13)

        if True:
            outputs[i*3+0] = dx
            outputs[i*3+1] =  dy*math.cos(theta) + dz*math.sin(theta)
            outputs[i*3+2] = -dy*math.sin(theta) + dz*math.cos(theta)

    plot(outputs,inputs)

# ======================
# 2d-pose Utils
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def line(input_img, person, point1, point2):
    threshold = 0.2
    if person.points[point1].score > threshold and\
       person.points[point2].score > threshold:
        color = hsv_to_rgb(255*point1/ailia.POSE_KEYPOINT_CNT, 255, 255)

        x1 = int(input_img.shape[1] * person.points[point1].x)
        y1 = int(input_img.shape[0] * person.points[point1].y)
        x2 = int(input_img.shape[1] * person.points[point2].x)
        y2 = int(input_img.shape[0] * person.points[point2].y)
        cv2.line(input_img, (x1, y1), (x2, y2), color, 5)


def display_result(input_img, pose, baseline):
    count = pose.get_object_count()
    if count >= 1:
        count = 1

    for idx in range(count):
        person = pose.get_object_pose(idx)

        line(input_img, person, ailia.POSE_KEYPOINT_NOSE,
                ailia.POSE_KEYPOINT_SHOULDER_CENTER)
        line(input_img, person, ailia.POSE_KEYPOINT_SHOULDER_LEFT,
                ailia.POSE_KEYPOINT_SHOULDER_CENTER)
        line(input_img, person, ailia.POSE_KEYPOINT_SHOULDER_RIGHT,
                ailia.POSE_KEYPOINT_SHOULDER_CENTER)

        line(input_img, person, ailia.POSE_KEYPOINT_EYE_LEFT,
                ailia.POSE_KEYPOINT_NOSE)
        line(input_img, person, ailia.POSE_KEYPOINT_EYE_RIGHT,
                ailia.POSE_KEYPOINT_NOSE)
        line(input_img, person, ailia.POSE_KEYPOINT_EAR_LEFT,
                ailia.POSE_KEYPOINT_EYE_LEFT)
        line(input_img, person, ailia.POSE_KEYPOINT_EAR_RIGHT,
                ailia.POSE_KEYPOINT_EYE_RIGHT)

        line(input_img, person, ailia.POSE_KEYPOINT_ELBOW_LEFT,
                ailia.POSE_KEYPOINT_SHOULDER_LEFT)
        line(input_img, person, ailia.POSE_KEYPOINT_ELBOW_RIGHT,
                ailia.POSE_KEYPOINT_SHOULDER_RIGHT)
        line(input_img, person, ailia.POSE_KEYPOINT_WRIST_LEFT,
                ailia.POSE_KEYPOINT_ELBOW_LEFT)
        line(input_img, person, ailia.POSE_KEYPOINT_WRIST_RIGHT,
                ailia.POSE_KEYPOINT_ELBOW_RIGHT)

        line(input_img, person, ailia.POSE_KEYPOINT_BODY_CENTER,
                ailia.POSE_KEYPOINT_SHOULDER_CENTER)
        line(input_img, person, ailia.POSE_KEYPOINT_HIP_LEFT,
                ailia.POSE_KEYPOINT_BODY_CENTER)
        line(input_img, person, ailia.POSE_KEYPOINT_HIP_RIGHT,
                ailia.POSE_KEYPOINT_BODY_CENTER)

        line(input_img, person, ailia.POSE_KEYPOINT_KNEE_LEFT,
                ailia.POSE_KEYPOINT_HIP_LEFT)
        line(input_img, person, ailia.POSE_KEYPOINT_ANKLE_LEFT,
                ailia.POSE_KEYPOINT_KNEE_LEFT)
        line(input_img, person, ailia.POSE_KEYPOINT_KNEE_RIGHT,
                ailia.POSE_KEYPOINT_HIP_RIGHT)
        line(input_img, person, ailia.POSE_KEYPOINT_ANKLE_RIGHT,
                ailia.POSE_KEYPOINT_KNEE_RIGHT)
        
        points=[]
        points.append(person.points[ailia.POSE_KEYPOINT_NOSE].x)    #OPENPOSE_Nose
        points.append(person.points[ailia.POSE_KEYPOINT_NOSE].y)
        points.append(person.points[ailia.POSE_KEYPOINT_SHOULDER_CENTER].x)    #OPENPOSE_Neck
        points.append(person.points[ailia.POSE_KEYPOINT_SHOULDER_CENTER].y)
        points.append(person.points[ailia.POSE_KEYPOINT_SHOULDER_RIGHT].x)    #OPENPOSE_RightShoulder
        points.append(person.points[ailia.POSE_KEYPOINT_SHOULDER_RIGHT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_ELBOW_RIGHT].x)    #OPENPOSE_RightElbow
        points.append(person.points[ailia.POSE_KEYPOINT_ELBOW_RIGHT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_WRIST_RIGHT].x)    #OPENPOSE_RightWrist
        points.append(person.points[ailia.POSE_KEYPOINT_WRIST_RIGHT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_SHOULDER_LEFT].x)    #OPENPOSE_LeftShoulder
        points.append(person.points[ailia.POSE_KEYPOINT_SHOULDER_LEFT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_ELBOW_LEFT].x)    #OPENPOSE_LeftElbow
        points.append(person.points[ailia.POSE_KEYPOINT_ELBOW_LEFT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_WRIST_LEFT].x)    #OPENPOSE_LeftWrist
        points.append(person.points[ailia.POSE_KEYPOINT_WRIST_LEFT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_HIP_RIGHT].x)    #OPENPOSE_RightHip
        points.append(person.points[ailia.POSE_KEYPOINT_HIP_RIGHT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_KNEE_RIGHT].x)    #OPENPOSE_RightKnee
        points.append(person.points[ailia.POSE_KEYPOINT_KNEE_RIGHT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_ANKLE_RIGHT].x)    #OPENPOSE_RightAnkle
        points.append(person.points[ailia.POSE_KEYPOINT_ANKLE_RIGHT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_HIP_LEFT].x)    #OPENPOSE_LeftHip
        points.append(person.points[ailia.POSE_KEYPOINT_HIP_LEFT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_KNEE_LEFT].x)    #OPENPOSE_LeftKnee
        points.append(person.points[ailia.POSE_KEYPOINT_KNEE_LEFT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_ANKLE_LEFT].x)    #OPENPOSE_LAnkle
        points.append(person.points[ailia.POSE_KEYPOINT_ANKLE_LEFT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_EYE_RIGHT].x)    #OPENPOSE_RightEye
        points.append(person.points[ailia.POSE_KEYPOINT_EYE_RIGHT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_EYE_LEFT].x)    #OPENPOSE_LeftEye
        points.append(person.points[ailia.POSE_KEYPOINT_EYE_LEFT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_EAR_RIGHT].x)    #OPENPOSE_RightEar
        points.append(person.points[ailia.POSE_KEYPOINT_EAR_RIGHT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_EYE_LEFT].x)    #OPENPOSE_LeftEar
        points.append(person.points[ailia.POSE_KEYPOINT_EYE_LEFT].y)
        points.append(person.points[ailia.POSE_KEYPOINT_BODY_CENTER].x)    #OPENPOSE_Background
        points.append(person.points[ailia.POSE_KEYPOINT_BODY_CENTER].y)

        display_3d_pose(points,baseline)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    src_img = cv2.imread(args.input)
    input_image = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None'
    )
    input_data = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGRA)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    pose = ailia.PoseEstimator(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id, algorithm=ALGORITHM
    )
    baseline = ailia.Net(
        BASELINE_MODEL_PATH,BASELINE_WEIGHT_PATH,env_id=env_id
    )
    baseline.set_input_shape((1,32))

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            _ = pose.compute(input_data)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        _ = pose.compute(input_data)

    # postprocessing
    count = pose.get_object_count()
    print(f'person_count={count}')
    display_result(src_img, pose, baseline)
    cv2.imwrite(args.savepath, src_img)
    print('Script finished successfully.')

    # display 3d pose
    plt.show()
    #fig = plt.figure()
    #fig.savefig("output_3dpose.png")

def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    pose = ailia.PoseEstimator(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id, algorithm=ALGORITHM
    )
    baseline = ailia.Net(
        BASELINE_MODEL_PATH,BASELINE_WEIGHT_PATH,env_id=env_id
    )
    baseline.set_input_shape((1,32))

    if args.video == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    else:
        if check_file_existance(args.video):
            capture = cv2.VideoCapture(args.video)

    while(True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        input_image, input_data = adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH,
        )
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2BGRA)

        # inferece
        _ = pose.compute(input_data)

        # postprocessing
        display_result(input_image, pose, baseline)
        cv2.imshow('frame', input_image)

        # display 3d pose
        plt.pause(0.01)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(BASELINE_WEIGHT_PATH, BASELINE_MODEL_PATH, BASELINE_REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
