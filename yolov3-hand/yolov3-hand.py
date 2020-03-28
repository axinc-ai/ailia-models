#ailia detector api sample

import ailia

import numpy
import tempfile
import cv2
import os
import urllib.request
import sys
import time

#require ailia SDK 1.2.1

MODE="image"
if len(sys.argv)>=2:
	MODE = sys.argv[1]
	if MODE!="image" and MODE!="video":
		print("please set mdoe to image or video")
		sys.exit()

model_path = "yolov3-hand.opt.onnx.prototxt"
weight_path = "yolov3-hand.opt.onnx"
img_path = "couple.jpg"

print("downloading ...");

if not os.path.exists(model_path):
	urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/yolov3-hand/"+model_path,model_path)
if not os.path.exists(weight_path):
	urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/yolov3-hand/"+weight_path,weight_path)

print("loading ...");

# detector initialize
env_id = ailia.get_gpu_environment_id()
categories = 80
threshold = 0.4
iou = 0.45
detector = ailia.Detector(model_path, weight_path, categories, format=ailia.NETWORK_IMAGE_FORMAT_RGB, channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST, range=ailia.NETWORK_IMAGE_RANGE_U_FP32, algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3, env_id=env_id)

# category
coco_category=[
	"hand"
]

def hsv_to_rgb(h, s, v):
	bgr = cv2.cvtColor(numpy.array([[[h, s, v]]], dtype=numpy.uint8), cv2.COLOR_HSV2BGR)[0][0]
	return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)

def display_result(work,detector,logging):
	# get result
	count = detector.get_object_count()

	if logging:
		print("object_count=" + str(count))

	w = work.shape[1]
	h = work.shape[0]

	for idx  in range(count) :
		# print result
		obj = detector.get_object(idx)
		if logging:
			print("+ idx=" + str(idx))
			print("  category=" + str(obj.category) + "[ " + coco_category[obj.category] + " ]" )
			print("  prob=" + str(obj.prob) )
			print("  x=" + str(obj.x) )
			print("  y=" + str(obj.y) )
			print("  w=" + str(obj.w) )
			print("  h=" + str(obj.h) )
		top_left = ( int(w*obj.x), int(h*obj.y) )
		bottom_right = ( int(w*(obj.x+obj.w)), int(h*(obj.y+obj.h)) )
		text_position = ( int(w*obj.x)+4, int(h*(obj.y+obj.h)-8) )

		# update image
		color = hsv_to_rgb(255*obj.category/80,255,255)
		cv2.rectangle( work, top_left, bottom_right, color, 4)
		fontScale=w/512.0
		cv2.putText( work, coco_category[obj.category], text_position, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, 1)

def recognize_from_image():
	# load input image and convert to BGRA
	img = cv2.imread( img_path, cv2.IMREAD_UNCHANGED )
	if img.shape[2] == 3 :
		img = cv2.cvtColor( img, cv2.COLOR_BGR2BGRA )
	elif img.shape[2] == 1 : 
		img = cv2.cvtColor( img, cv2.COLOR_GRAY2BGRA )

	print( "img.shape=" + str(img.shape) )

	work = img
	w = img.shape[1]
	h = img.shape[0]

	print("inferencing ...");

	# compute

	cnt = 3
	for i in range(cnt):
		start=int(round(time.time() * 1000))
		detector.compute(img, threshold, iou)
		end=int(round(time.time() * 1000))
		print("## ailia processing time , "+str(i)+" , "+str(end-start)+" ms")

	display_result(work,detector,True)

	# save image
	cv2.imwrite( "output.png", work)

def recognize_from_video():
	capture = cv2.VideoCapture(0)
	if not capture.isOpened():
		print("webcamera not found")
		sys.exit()
	while(True):
		ret, frame = capture.read()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if not ret:
			continue

		input_img = numpy.array(frame)
		img = cv2.cvtColor( frame, cv2.COLOR_BGR2BGRA )
		detector.compute(img, threshold, iou)
		display_result(input_img,detector,False)

		cv2.imshow('frame',input_img)
	capture.release()
	cv2.destroyAllWindows()

if MODE=="image":
	recognize_from_image()

if MODE=="video":
	recognize_from_video()