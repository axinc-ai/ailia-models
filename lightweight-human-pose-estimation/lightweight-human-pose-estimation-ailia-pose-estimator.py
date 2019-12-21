import requests
import numpy as np
import time
import os
import cv2
import urllib.request

import ailia
import ailia_pose_estimator

#require ailia SDK 1.2.1

OPT_MODEL=True
if OPT_MODEL:
	model_path = "lightweight-human-pose-estimation.opt.onnx.prototxt"
	weight_path = "lightweight-human-pose-estimation.opt.onnx"
else:
	model_path = "lightweight-human-pose-estimation.onnx.prototxt"
	weight_path = "lightweight-human-pose-estimation.onnx"

print("downloading ...");

if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/lightweight-human-pose-estimation/"+model_path,model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/lightweight-human-pose-estimation/"+weight_path,weight_path)

print("loading ...");

algorithm = ailia_pose_estimator.ALGORITHM_LW_HUMAN_POSE

ailia_input_width = 320
ailia_input_height = 240

env_id=ailia.ENVIRONMENT_AUTO

file_name = 'balloon.png'

img = cv2.imread(file_name)
input_img = np.array(img)
img = cv2.resize(img,(ailia_input_width,ailia_input_height))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

env_id=ailia.get_gpu_environment_id()
net = ailia.Net(model_path,weight_path,env_id=env_id)
pose = ailia_pose_estimator.PoseEstimator(net,algorithm)

print("inferencing ...");

cnt = 3
for i in range(cnt):
	start=int(round(time.time() * 1000))
	persons = pose.compute(img)
	end=int(round(time.time() * 1000))
	print("## ailia processing time , "+str(i)+" , "+str(end-start)+" ms")

if(not persons):
	print("person not detected")

for person in persons:
	print("person detected");
	for i in range(ailia_pose_estimator.POSE_KEYPOINT_CNT):
		score = person.points[i].score
		x = (input_img.shape[1] * person.points[i].x)
		y = (input_img.shape[0] * person.points[i].y)
		threshold=0.2
		if person.points[i].score > threshold :
			cv2.circle(input_img, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
			cv2.putText(input_img, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, lineType=cv2.LINE_AA)
			cv2.putText(input_img, ""+str(score), (int(x), int(y+16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

cv2.imshow("Keypoints",input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
