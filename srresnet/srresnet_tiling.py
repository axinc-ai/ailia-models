import numpy as np
import time
import os
import cv2
import urllib.request
import sys

from PIL import Image

import ailia

if len(sys.argv) >= 3:
  INPUT_IMAGE_PATH=sys.argv[1]
  OUTPUT_IMAGE_PATH=sys.argv[2]
else:
  print("usage: python srresnet_tiling.py input.png output.png")
  sys.exit(1)

model_path = "srresnet.onnx.prototxt"
weight_path = "srresnet.onnx"

if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/srresnet/"+model_path,model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/srresnet/"+weight_path,weight_path)

env_id=ailia.get_gpu_environment_id()
net = ailia.Net(model_path,weight_path,env_id=env_id)

ailia_input_width = net.get_input_shape()[3]
ailia_input_height = net.get_input_shape()[2]

ailia_output_width = net.get_output_shape()[3]
ailia_output_height = net.get_output_shape()[2]

img = cv2.imread(INPUT_IMAGE_PATH)

input_width = img.shape[1]
input_height = img.shape[0]

padding_width = int((input_width+ailia_input_width-1)/ailia_input_width)*ailia_input_width
padding_height = int((input_height+ailia_input_height-1)/ailia_input_height)*ailia_input_height

scale = int(ailia_output_height/ailia_input_height)

output_padding_width = padding_width*scale
output_padding_height = padding_height*scale

output_width = input_width*scale
output_height = input_height*scale

print("input image : "+str(input_width)+"x"+str(input_height))
print("output image : "+str(output_width)+"x"+str(output_height))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img.shape = (1,) + img.shape
img = img.transpose((0, 3, 1, 2))
img = np.array(img)
img = img.astype(np.float32)
img = img / 255.0

padding_img=np.zeros((1,3,padding_height,padding_width))
padding_img[:,:,0:input_height,0:input_width]=img

output_padding_img=np.zeros((1,3,output_padding_height,output_padding_width))

start = int(round(time.time() * 1000))
tile_x = int(padding_width/ailia_input_width)
tile_y = int(padding_height/ailia_input_height)
for y in range(tile_y):
	for x in range(tile_x):
		output_tile = net.predict(padding_img[:,:,y*ailia_input_height:(y+1)*ailia_input_height,x*ailia_input_width:(x+1)*ailia_input_width])
		output_padding_img[:,:,y*ailia_output_height:(y+1)*ailia_output_height,x*ailia_output_width:(x+1)*ailia_output_width] = output_tile[:,:,:,:]
end = int(round(time.time() * 1000))
print("ailia processing time , "+str(x)+" ,"+str(y)+" , "+str(end-start)+" ms")

output_img = output_padding_img[:,:,0:output_height,0:output_width]
output_img = output_img.transpose((0,2,3,1))
shape = output_img.shape
output_img = output_img.reshape((shape[1],shape[2],shape[3]))
output_img = output_img*255.
output_img[output_img<0] = 0
output_img[output_img>255.] = 255.            
output_img = output_img.astype(np.int8)
img2 = Image.fromarray(output_img, 'RGB')
img2.save(OUTPUT_IMAGE_PATH)
