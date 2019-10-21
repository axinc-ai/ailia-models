import cv2
import sys
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import urllib.request

import ailia

model_path = "lightweight-human-pose-estimation.onnx.prototxt"
weight_path = "lightweight-human-pose-estimation.onnx"

print("downloading ...");

if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/lightweight-human-pose-estimation/"+model_path,model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/lightweight-human-pose-estimation/"+weight_path,weight_path)

print("loading ...");

env_id=0
env_id=ailia.get_gpu_environment_id()
net = ailia.Net(model_path,weight_path,env_id=env_id)

IMAGE_PATH="balloon.png"

IMAGE_WIDTH=net.get_input_shape()[3]
IMAGE_HEIGHT=net.get_input_shape()[2]

input_img = cv2.imread(IMAGE_PATH)

img = cv2.resize(input_img, (IMAGE_WIDTH, IMAGE_HEIGHT))

img = img[...,::-1]  #BGR 2 RGB

data = np.array(img, dtype=np.float32)
data.shape = (1,) + data.shape
data = data / 255.0
data = data.transpose((0, 3, 1, 2))

print("inferencing ...");

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

print("PAF SHAPE : "+str(paf.shape))
print("CONFIDENCE SHAPE : "+str(confidence.shape))

points = []
threshold = 0.1

for i in range(confidence.shape[1]):
	probMap = confidence[0, i, :, :]
	minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

	x = (input_img.shape[1] * point[0]) / confidence.shape[3]
	y = (input_img.shape[0] * point[1]) / confidence.shape[2]
 
	if prob > threshold : 
		circle_size = 4
		cv2.circle(input_img, (int(x), int(y)), circle_size, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
		#cv2.putText(input_img, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, lineType=cv2.LINE_AA)
		#cv2.putText(input_img, ""+str(prob), (int(x), int(y+circle_size)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

		points.append((int(x), int(y)))
	else :
		points.append(None)
 
cv2.imshow("Keypoints",input_img)
cv2.imwrite('output.png', input_img)

def plot_images(title, images, tile_shape):
	assert images.shape[0] <= (tile_shape[0]* tile_shape[1])
	from mpl_toolkits.axes_grid1 import ImageGrid
	fig = plt.figure()
	plt.title(title)
	grid = ImageGrid(fig, 111,  nrows_ncols = tile_shape )
	for i in range(images.shape[1]):
		grd = grid[i]
		grd.imshow(images[0,i])

channels=max(confidence.shape[1],paf.shape[1])
cols=8

plot_images("paf",paf,tile_shape=((int)((channels+cols-1)/cols),cols))
plot_images("confidence",confidence,tile_shape=((int)((channels+cols-1)/cols),cols))

plt.show()

cv2.destroyAllWindows()
