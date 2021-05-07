# Normalize action keypoint
# (c) 2020 ax Inc.

import ailia
import math
import numpy as np

TIME_RANGE = 15

def pose_postprocess(pose_keypoints):
    thre = 0.2
    pose_keypoints[:, :, 0:2] = pose_keypoints[:, :, 0:2] - 0.5
    # pose_keypoints[:, :, 0][pose_keypoints[:, :, 2] == 0] = 0
    # pose_keypoints[:, :, 1][pose_keypoints[:, :, 2] == 0] = 0
    pose_keypoints[:, :, 0][pose_keypoints[:, :, 2] < thre] = 0
    pose_keypoints[:, :, 1][pose_keypoints[:, :, 2] < thre] = 0
    pose_keypoints[:, :, 2][pose_keypoints[:, :, 2] < thre] = 0
    return pose_keypoints
    
def push_keypoint(keypoints):
    frame = np.zeros((ailia.POSE_KEYPOINT_CNT,1,3))
    for i in range(0,ailia.POSE_KEYPOINT_CNT):
        frame[i,0,0]=keypoints[i].x
        frame[i,0,1]=keypoints[i].y
        frame[i,0,2]=keypoints[i].score
    return frame

def get_vector(keypoints, from_point, to_point, j):
    x1 = keypoints[from_point,j,0]
    y1 = keypoints[from_point,j,1]
    c1 = keypoints[from_point,j,2]
    x2 = keypoints[to_point,j,0]
    y2 = keypoints[to_point,j,1]
    c2 = keypoints[to_point,j,2]
    thre = 0.2
    if c1<thre or c2<thre:
        return 0,0,0
    r = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    # return (x1-x2)/r,(y1-y2)/r
    # return (x1-x2)/r,(y1-y2)/r,r
    atan = math.atan2(y2-y1, x2-x1)
    atan_scaled = max(min((atan/math.pi), 1), -1)
    c = 2*(c1*c2-0.5)
    return atan_scaled,r,c



def normalize_keypoint(keypoints, threshold=0.3):
    #normalize points
    frame = keypoints.copy()
    for j in range(TIME_RANGE):
        # neck = frame[ailia.POSE_KEYPOINT_SHOULDER_CENTER,j]
        for i in range(0,ailia.POSE_KEYPOINT_CNT):
            v1=0
            v2=0
            r=0
            c=0
            if i==0:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_HIP_LEFT,ailia.POSE_KEYPOINT_KNEE_LEFT, j)
            if i==1:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_KNEE_LEFT,ailia.POSE_KEYPOINT_ANKLE_LEFT, j)
            if i==2:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_HIP_RIGHT,ailia.POSE_KEYPOINT_KNEE_RIGHT, j)
            if i==3:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_KNEE_RIGHT,ailia.POSE_KEYPOINT_ANKLE_RIGHT, j)
            if i==4:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_SHOULDER_LEFT,ailia.POSE_KEYPOINT_ELBOW_LEFT, j)
            if i==5:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_ELBOW_LEFT,ailia.POSE_KEYPOINT_WRIST_LEFT, j)
            if i==6:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_SHOULDER_RIGHT,ailia.POSE_KEYPOINT_ELBOW_RIGHT, j)
            if i==7:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_ELBOW_RIGHT,ailia.POSE_KEYPOINT_WRIST_RIGHT, j)
            if i==8:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_SHOULDER_LEFT,ailia.POSE_KEYPOINT_SHOULDER_RIGHT, j)
            if i==9:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_SHOULDER_LEFT,ailia.POSE_KEYPOINT_HIP_LEFT, j)
            if i==10:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_SHOULDER_RIGHT,ailia.POSE_KEYPOINT_HIP_RIGHT, j)
            if i==11:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_SHOULDER_CENTER,ailia.POSE_KEYPOINT_NOSE, j)
            if i==12:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_NOSE,ailia.POSE_KEYPOINT_EYE_LEFT, j)
            if i==13:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_NOSE,ailia.POSE_KEYPOINT_EYE_RIGHT, j)
            if i==14:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_EAR_LEFT,ailia.POSE_KEYPOINT_EYE_LEFT, j)
            if i==15:
                v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_EAR_RIGHT,ailia.POSE_KEYPOINT_EYE_RIGHT, j)

            # if i==8:
            #     v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_SHOULDER_CENTER,ailia.POSE_KEYPOINT_BODY_CENTER, j)
            # if i==9:
            #     v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_BODY_CENTER,ailia.POSE_KEYPOINT_HIP_LEFT, j)
            # if i==10:
            #     v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_BODY_CENTER,ailia.POSE_KEYPOINT_HIP_RIGHT, j)  
            # if i==11:
            #     v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_SHOULDER_CENTER,ailia.POSE_KEYPOINT_NOSE, j)
            # if i==12:
            #     v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_NOSE,ailia.POSE_KEYPOINT_EAR_LEFT, j)
            # if i==13:
            #     v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_NOSE,ailia.POSE_KEYPOINT_EAR_RIGHT, j)
            # if i==14:
            #     v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_SHOULDER_LEFT,ailia.POSE_KEYPOINT_SHOULDER_RIGHT, j)
            # if i==15:
            #     v1,r,c = get_vector(keypoints,ailia.POSE_KEYPOINT_HIP_LEFT,ailia.POSE_KEYPOINT_HIP_RIGHT, j)
 
            frame[i,j,0]=int(min(max(0,v1*127+128),255))
            frame[i,j,1]=r
            frame[i,j,2]=int(min(max(0,c*127+128),255))

    r_max = np.amax(frame[:,:,1])
    if r_max == 0:
        frame[:,:,1]=0
        return frame[:,:,:]
    # frame[:,:,1] = int(min(max(0,r/r_max*255),255))
    frame[:,:,1] = frame[:,:,1]/r_max*255
    frame.astype(int)

    return frame[:,:,:]