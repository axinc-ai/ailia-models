#ailia pose estimator api sample

import requests
import numpy as np
import time
import os
import cv2
import urllib

import ailia

#OPENPOSE: MULTIPERSON KEYPOINT DETECTION
#SOFTWARE LICENSE AGREEMENT
#ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

model_path = "pose_deploy.prototxt"
weight_path = "pose_iter_440000.caffemodel"

print("downloading ...");

if not os.path.exists(weight_path):
    urllib.request.urlretrieve("http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/"+weight_path,weight_path)

algorithm = ailia.POSE_ALGORITHM_OPEN_POSE

file_name = 'balloon.png'

# estimator initialize
env_id=ailia.get_gpu_environment_id()
pose = ailia.PoseEstimator(model_path,weight_path, env_id=env_id, algorithm=algorithm)

shape = pose.get_input_shape()
ailia_input_width = shape[3]
ailia_input_height = shape[2]

# load input image and convert to BGRA
img = cv2.imread(file_name)
input_img = np.array(img)
img = cv2.resize(img,(ailia_input_width,ailia_input_height))
if img.shape[2] == 3 :
    img = cv2.cvtColor( img, cv2.COLOR_BGR2BGRA )
elif img.shape[2] == 1 : 
    img = cv2.cvtColor( img, cv2.COLOR_GRAY2BGRA )

# compute
cnt = 3
for i in range(cnt):
	start=int(round(time.time() * 1000))
	persons = pose.compute(img)
	end=int(round(time.time() * 1000))
	print("## ailia processing time , "+str(i)+" , "+str(end-start)+" ms")

count = pose.get_object_count()

print("person_count=" + str(count))

# print result
def hsv_to_rgb(h, s, v):
	bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
	return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

def line(input_img,person,point1,point2):
	threshold = 0.2
	if person.points[point1].score>threshold and person.points[point2].score>threshold:
		color = hsv_to_rgb(255*point1/ailia.POSE_KEYPOINT_CNT,255,255)

		x1 = int(input_img.shape[1] * person.points[point1].x)
		y1 = int(input_img.shape[0] * person.points[point1].y)

		x2 = int(input_img.shape[1] * person.points[point2].x)
		y2 = int(input_img.shape[0] * person.points[point2].y)

		cv2.line(input_img,(x1,y1),(x2,y2),color,5)


for idx  in range(count) :
	person = pose.get_object_pose(idx)
	for i in range(ailia.POSE_KEYPOINT_CNT):
		score = person.points[i].score
		x = (input_img.shape[1] * person.points[i].x)
		y = (input_img.shape[0] * person.points[i].y)

		line(input_img,person,ailia.POSE_KEYPOINT_NOSE,ailia.POSE_KEYPOINT_SHOULDER_CENTER)
		line(input_img,person,ailia.POSE_KEYPOINT_SHOULDER_LEFT,ailia.POSE_KEYPOINT_SHOULDER_CENTER)
		line(input_img,person,ailia.POSE_KEYPOINT_SHOULDER_RIGHT,ailia.POSE_KEYPOINT_SHOULDER_CENTER)

		line(input_img,person,ailia.POSE_KEYPOINT_EYE_LEFT,ailia.POSE_KEYPOINT_NOSE)
		line(input_img,person,ailia.POSE_KEYPOINT_EYE_RIGHT,ailia.POSE_KEYPOINT_NOSE)
		line(input_img,person,ailia.POSE_KEYPOINT_EAR_LEFT,ailia.POSE_KEYPOINT_EYE_LEFT)
		line(input_img,person,ailia.POSE_KEYPOINT_EAR_RIGHT,ailia.POSE_KEYPOINT_EYE_RIGHT)

		line(input_img,person,ailia.POSE_KEYPOINT_ELBOW_LEFT,ailia.POSE_KEYPOINT_SHOULDER_LEFT)
		line(input_img,person,ailia.POSE_KEYPOINT_ELBOW_RIGHT,ailia.POSE_KEYPOINT_SHOULDER_RIGHT)
		line(input_img,person,ailia.POSE_KEYPOINT_WRIST_LEFT,ailia.POSE_KEYPOINT_ELBOW_LEFT)
		line(input_img,person,ailia.POSE_KEYPOINT_WRIST_RIGHT,ailia.POSE_KEYPOINT_ELBOW_RIGHT)

		line(input_img,person,ailia.POSE_KEYPOINT_BODY_CENTER,ailia.POSE_KEYPOINT_SHOULDER_CENTER)
		line(input_img,person,ailia.POSE_KEYPOINT_HIP_LEFT,ailia.POSE_KEYPOINT_BODY_CENTER)
		line(input_img,person,ailia.POSE_KEYPOINT_HIP_RIGHT,ailia.POSE_KEYPOINT_BODY_CENTER)

		line(input_img,person,ailia.POSE_KEYPOINT_KNEE_LEFT,ailia.POSE_KEYPOINT_HIP_LEFT)
		line(input_img,person,ailia.POSE_KEYPOINT_ANKLE_LEFT,ailia.POSE_KEYPOINT_KNEE_LEFT)
		line(input_img,person,ailia.POSE_KEYPOINT_KNEE_RIGHT,ailia.POSE_KEYPOINT_HIP_RIGHT)
		line(input_img,person,ailia.POSE_KEYPOINT_ANKLE_RIGHT,ailia.POSE_KEYPOINT_KNEE_RIGHT)

# save image
cv2.imwrite( "output.png", input_img)
