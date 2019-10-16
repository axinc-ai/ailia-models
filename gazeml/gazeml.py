import cv2
import sys
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request

import ailia

#model_path = "gazeml_elg_i180x108_n64.onnx.prototxt"
#weight_path = "gazeml_elg_i180x108_n64.onnx"

model_path = "gazeml_elg_i60x36_n32.onnx.prototxt"
weight_path = "gazeml_elg_i60x36_n32.onnx"

print("downloading ...");

if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/gazeml/"+model_path,model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/gazeml/"+weight_path,weight_path)

print("loading ...");

env_id=ailia.get_gpu_environment_id()
net = ailia.Net(model_path,weight_path,env_id=env_id)

IMAGE_PATH="eye.png"

IMAGE_WIDTH=net.get_input_shape()[2]
IMAGE_HEIGHT=net.get_input_shape()[1]

input_img = cv2.imread(IMAGE_PATH)

img = cv2.resize(input_img, (IMAGE_WIDTH, IMAGE_HEIGHT))

img = img[...,::-1]  #BGR 2 RGB

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img = cv2.equalizeHist(img)
cv2.imshow("equalize",img)

data = np.array(img, dtype=np.float32)
data.shape = (1,1,) + data.shape
data = data / 255.0 *2 -1.0

print("inferencing ...");

eyeI = np.concatenate((data, data), axis=0)
eyeI = eyeI.reshape(2,IMAGE_HEIGHT,IMAGE_WIDTH,1)
pred_onnx = net.predict(eyeI)
out = pred_onnx

cnt = 3
for i in range(cnt):
	start=int(round(time.time() * 1000))
	out = net.predict(eyeI)
	end=int(round(time.time() * 1000))
	print("## ailia processing time , "+str(i)+" , "+str(end-start)+" ms")

points = []
threshold = 0.1
scale = 1.2

for i in range(out.shape[3]):
	probMap = out[0, :, :, i]
	minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

	x = (input_img.shape[1] * point[0]) / out.shape[2] * scale
	y = (input_img.shape[0] * point[1]) / out.shape[1] * scale

	color = (0, 255, 255)
	if i>=8:
		color = (255, 0, 0)
	if i>=16:
		color = (0, 0, 255)

	if prob > threshold : 
		cv2.circle(input_img, (int(x), int(y)), 3, color, thickness=-1, lineType=cv2.FILLED)
		points.append((int(x), int(y)))
	else :
		points.append(None)
	
cv2.imshow("Keypoints",input_img)
cv2.imwrite('output.png', input_img)

def plot_images(title, images, tile_shape):
	from mpl_toolkits.axes_grid1 import ImageGrid
	fig = plt.figure()
	plt.title(title)
	grid = ImageGrid(fig, 111,  nrows_ncols = tile_shape )
	for i in range(images.shape[3]):
		grd = grid[i]
		grd.imshow(images[0,:,:,i])

channels=out.shape[3]
cols=8

plot_images("result",out,tile_shape=((int)((channels+cols-1)/cols),cols))
plt.show()

cv2.destroyAllWindows()
