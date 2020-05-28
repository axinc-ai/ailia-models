import cv2
import sys
import numpy as np
import pandas as pd
import os
import time
import ailia
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import urllib.request
import math

from mpl_toolkits.mplot3d import Axes3D

ONNX_RUNTIME=False
KERAS=False
model_path = "3d-pose-baseline.onnx.prototxt"
weight_path = "3d-pose-baseline.onnx"

POSE_ESTIMATION=True
pose_model_path = "lightweight-human-pose-estimation.onnx.prototxt"
pose_weight_path = "lightweight-human-pose-estimation.onnx"

print("downloading ...");

if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/3d-pose-baseline/"+model_path,model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/3d-pose-baseline/"+weight_path,weight_path)
if POSE_ESTIMATION:
	if not os.path.exists(pose_model_path):
		urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/lightweight-human-pose-estimation/"+pose_model_path,pose_model_path)
	if not os.path.exists(pose_weight_path):
		urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/lightweight-human-pose-estimation/"+pose_weight_path,pose_weight_path)

print("estimate 2d pose ...");

if POSE_ESTIMATION:
	IMAGE_PATH = "running.jpg"

	if not os.path.exists(IMAGE_PATH):
		print(IMAGE_PATH+" not found")
		sys.exit(1)

	env_id=0
	env_id=ailia.get_gpu_environment_id()
	net = ailia.Net(pose_model_path,pose_weight_path,env_id=env_id)

	IMAGE_WIDTH=net.get_input_shape()[3]
	IMAGE_HEIGHT=net.get_input_shape()[2]

	input_img = cv2.imread(IMAGE_PATH)

	img = cv2.resize(input_img, (IMAGE_WIDTH, IMAGE_HEIGHT))

	img = img[...,::-1]  #BGR 2 RGB

	data = np.array(img, dtype=np.float32)
	data.shape = (1,) + data.shape
	data = data / 255.0
	data = data.transpose((0, 3, 1, 2))

	pred_onnx = net.predict(data)
	out = pred_onnx

	cnt = 3
	for i in range(cnt):
		if(i==1):
			net.set_profile_mode()
		start=int(round(time.time() * 1000))
		out = net.predict(data)
		end=int(round(time.time() * 1000))
		print("## ailia processing time , "+str(i)+" , "+str(end-start)+" ms")

	confidence = net.get_blob_data(net.find_blob_index_by_name("397"))
	paf = net.get_blob_data(net.find_blob_index_by_name("400"))

	points = []
	threshold = 0.1

	for i in range(confidence.shape[1]):
		probMap = confidence[0, i, :, :]
		minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

		target_width = input_img.shape[1]
		target_height = input_img.shape[0]

		#target_width = 100
		#target_height = target_width * input_img.shape[0] / input_img.shape[1]

		x = (target_width * point[0]) / confidence.shape[3]
		y = (target_height * point[1]) / confidence.shape[2] 
	
		if prob > threshold : 
			points.append(x)
			points.append(y)
		else :
			points.append(0)
			points.append(0)

print("preparing 3d pose estimation ...");

with h5py.File('3d-pose-baseline-mean.h5', 'r') as f:
  data_mean_2d = np.array(f['data_mean_2d'])
  data_std_2d = np.array(f['data_std_2d'])
  data_mean_3d = np.array(f['data_mean_3d'])
  data_std_3d = np.array(f['data_std_3d'])

if not POSE_ESTIMATION:
	with h5py.File('3d-pose-baseline-test.hdf5', 'r') as f:
		enc_in = np.array(f['enc_in'])
		dec_out = np.array(f['dec_out'])
		poses3d = np.array(f['poses3d'])
		data_mean_2d_onnx = np.array(f['data_mean_2d'])
		data_std_2d_onnx = np.array(f['data_std_2d'])
		data_mean_3d_onnx = np.array(f['data_mean_3d'])
		data_std_3d_onnx = np.array(f['data_std_3d'])

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

if POSE_ESTIMATION:
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
		inputs[i*2+0]=(inputs[i*2+0]-data_mean_2d[j*2+0])/data_std_2d[j*2+0]
		inputs[i*2+1]=(inputs[i*2+1]-data_mean_2d[j*2+1])/data_std_2d[j*2+1]

print("predict 3d pose ...");

if not POSE_ESTIMATION:
	inputs=enc_in[0]
reshape_input = np.reshape(np.array(inputs),(1,32))

if ONNX_RUNTIME:
	import numpy
	import onnxruntime as rt

	onnx_sess = rt.InferenceSession(weight_path)

	node_name=""
	node_shape=(1,1)
	for node in onnx_sess.get_inputs():
		node_name=node.name
		node_shape=node.shape
		print(node)
	node_shape=(1,32)

	img = numpy.random.random(node_shape).astype(numpy.float32)

	start=int(round(time.time() * 1000))
	reshape_input = reshape_input.astype(numpy.float32)
	outputs = onnx_sess.run(None, {node_name:reshape_input})[0].reshape((48))
	end=int(round(time.time() * 1000))
	print("## onnxruntime processing time , "+str(0)+" , "+str(end-start)+" ms")
else:
	if KERAS:
		import keras
		from keras.models import load_model
		keras_model = load_model("3d-pose-baseline.hdf5")
		outputs = keras_model.predict(reshape_input,batch_size=1)[0]
	else:
		env_id=ailia.ENVIRONMENT_AUTO
		env_id=ailia.get_gpu_environment_id()
		net = ailia.Net(model_path,weight_path,env_id=env_id)
		net.set_input_shape((1,32))
		print(reshape_input.shape)
		outputs = net.predict(reshape_input)[0]
		print(outputs.shape)

print("display 3d pose ...");

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

	if ONNX_RUNTIME:
		outputs[i*3+0] = dx
		outputs[i*3+1] =  dy*math.cos(theta) + dz*math.sin(theta)
		outputs[i*3+2] = -dy*math.sin(theta) + dz*math.cos(theta)

fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(18, -70)  

IS_3D=False
cnt=0

X=[]
Y=[]
Z=[]

def search_name(name):
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

def draw_connect(from_id,to_id,color="#00aa00"):
	from_id=search_name(from_id)
	to_id=search_name(to_id)
	if(from_id==-1 or to_id==-1):
		return
	x = [X[from_id], X[to_id]]
	y = [Y[from_id], Y[to_id]]
	z = [Z[from_id], Z[to_id]]

	ax.plot(x, z, y, "o-", color=color, ms=4, mew=0.5)

def plot(data):
	plt.cla()

	ax.set_xlabel('X axis')
	ax.set_ylabel('Z axis')
	ax.set_zlabel('Y axis')
	ax.set_zlim([600, -600])

	global cnt,X,Y,Z,IS_3D
	k=cnt

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
			draw_connect("Head","Thorax","#0000aa")
			draw_connect("Thorax",'RShoulder')
			draw_connect('RShoulder','RElbow')
			draw_connect('RElbow','RWrist')
			draw_connect("Thorax",'LShoulder')
			draw_connect('LShoulder','LElbow')
			draw_connect('LElbow','LWrist')
			draw_connect('Thorax','Spine')
			draw_connect('Spine','LHip')
			draw_connect('Spine','RHip')
			draw_connect('RHip','RKnee')
			draw_connect('RKnee','RFoot')
			draw_connect('LHip','LKnee')
			draw_connect('LKnee','LFoot')
		else:
			draw_connect("Head","Thorax","#0000ff")
			draw_connect("Thorax",'RShoulder',"#00ff00")
			draw_connect('RShoulder','RElbow',"#00ff00")
			draw_connect('RElbow','RWrist',"#00ff00")
			draw_connect("Thorax",'LShoulder',"#00ff00")
			draw_connect('LShoulder','LElbow',"#00ff00")
			draw_connect('LElbow','LWrist',"#00ff00")
			draw_connect('Thorax','Spine',"#00ff00")
			draw_connect('Spine','Hip',"#00ff00")
			draw_connect('Hip','LHip',"#ff0000")
			draw_connect('Hip','RHip',"#ff0000")
			draw_connect('RHip','RKnee',"#ff0000")
			draw_connect('RKnee','RFoot',"#ff0000")
			draw_connect('LHip','LKnee',"#ff0000")
			draw_connect('LKnee','LFoot',"#ff0000")

ani = animation.FuncAnimation(fig, plot, interval=1000)
plt.show()
