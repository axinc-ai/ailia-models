import sys
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import adjust_frame_size  # noqa: E402C


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'couple.jpg'
SAVE_IMAGE_PATH = 'annotated.png'
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
MODEL_LISTS = ['mb1-ssd', 'mb2-ssd-lite']


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='MultiBox Detector'
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
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='mb2-ssd-lite', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS) + ' (default: mb2-ssd-lite)'
)
args = parser.parse_args()


# ======================
# Parameters 2
# ======================
WEIGHT_PATH = args.arch + '.onnx'
MODEL_PATH = args.arch + '.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mobilenet_ssd/'


# ======================
# Display result
# ======================
voc_category=[
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

def hsv_to_rgb(h, s, v):
	bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
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
			print("  category=" + str(obj.category) + "[ " + voc_category[obj.category] + " ]" )
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
		cv2.putText( work, voc_category[obj.category], text_position, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, 1)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    org_img = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None',
    )
    if org_img.shape[2] == 3 :
        org_img = cv2.cvtColor( org_img, cv2.COLOR_RGB2BGRA )

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    categories = 80
    threshold = 0.4
    iou = 0.45
    detector = ailia.Detector(MODEL_PATH, WEIGHT_PATH, categories, format=ailia.NETWORK_IMAGE_FORMAT_RGB, channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST, range=ailia.NETWORK_IMAGE_RANGE_U_FP32, algorithm=ailia.DETECTOR_ALGORITHM_SSD, env_id=env_id)

    # compute execution time
    for i in range(5):
        start = int(round(time.time() * 1000))
        detector.compute(org_img, threshold, iou)
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')

    # postprocessing
    display_result(org_img,detector,True)
    cv2.imwrite(args.savepath,org_img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    categories = 80
    threshold = 0.4
    iou = 0.45
    detector = ailia.Detector(MODEL_PATH, WEIGHT_PATH, categories, format=ailia.NETWORK_IMAGE_FORMAT_RGB, channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST, range=ailia.NETWORK_IMAGE_RANGE_U_FP32, algorithm=ailia.DETECTOR_ALGORITHM_SSD, env_id=env_id)

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

        _, resized_img = adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)

        img = cv2.cvtColor( resized_img, cv2.COLOR_RGB2BGRA )
        detector.compute(img, threshold, iou)
        display_result(resized_img,detector,False)

        cv2.imshow('frame', resized_img)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
