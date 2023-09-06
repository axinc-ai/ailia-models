from os import listdir, path
import numpy as np
import cv2, argparse, audio
import dlib, subprocess
from tqdm import tqdm

# Import original modules
import sys
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # NOQA: E402
from webcamera_utils import adjust_frame_size, get_capture, cut_max_square  # NOQA: E402

# Logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = "lipgan.onnx"
MODEL_PATH = "lipgan.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/psgan/"

FACE_DETECTOR_WEIGHT_PATH = "../../face_detection/blazeface/blazeface.onnx"
FACE_DETECTOR_MODEL_PATH = "../../face_detection/blazeface/blazeface.onnx.prototxt"
FACE_DETECTOR_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"

DLIB_WEIGHT_PATH = "mmod_human_face_detector.dat"

INPUT_IMAGE_PATH = "input.png"
INPUT_AUDIO_PATH = "input.wav"

SAVE_IMAGE_PATH = "./"

IMG_SIZE = 96

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
	"LipGAN",
	INPUT_IMAGE_PATH,
	SAVE_IMAGE_PATH,
)
parser.add_argument(
	"--audio", default=INPUT_AUDIO_PATH, help="Path to input audio"
)
parser.add_argument(
	"--onnx",
	action="store_true",
	help="Execute Onnx Runtime mode.",
)
parser.add_argument(
	"--use_dlib",
	action="store_true",
	help="Use dlib models for inference.",
)
args = update_parser(parser)


# ======================
# Functions
# ======================

# If video, until how many seconds of the clip to use for inference?
max_sec = 240.

# FPS of input video, ignore if image
fps = 25

# Padding (top, bottom, left, right)
pads = [0, 0, 0, 0]

# Single GPU batch size for face detection
face_det_batch_size = 64

# Single GPU batch size for LipGAN
lipgan_batch_size = 256

import ailia
ailia_net = ailia.Net(weight="lipgan.onnx", env_id = 2)

def rect_to_bb(d):
	x = d.rect.left()
	y = d.rect.top()
	w = d.rect.right() - x
	h = d.rect.bottom() - y
	return (x, y, w, h)

def calcMaxArea(rects):
	max_cords = (-1,-1,-1,-1)
	max_area = 0
	max_rect = None
	for i in range(len(rects)):
		cur_rect = rects[i]
		(x,y,w,h) = rect_to_bb(cur_rect)
		if w*h > max_area:
			max_area = w*h
			max_cords = (x,y,w,h)
			max_rect = cur_rect	
	return max_cords, max_rect
	
def face_detect(images):
	detector = dlib.cnn_face_detection_model_v1(DLIB_WEIGHT_PATH)

	batch_size = face_det_batch_size

	predictions = []
	for i in tqdm(range(0, len(images), batch_size)):
		predictions.extend(detector(images[i:i + batch_size]))
	
	results = []
	pady1, pady2, padx1, padx2 = pads
	for rects, image in zip(predictions, images):
		(x, y, w, h), max_rect = calcMaxArea(rects)
		if x == -1:
			results.append([None, (-1,-1,-1,-1), False])
			continue
		y1 = max(0, y + pady1)
		y2 = min(image.shape[0], y + h + pady2)
		x1 = max(0, x + padx1)
		x2 = min(image.shape[1], x + w + padx2)
		face = image[y1:y2, x1:x2, ::-1] # RGB ---> BGR

		results.append([face, (y1, y2, x1, x2), True])
	
	del detector # make sure to clear GPU memory for LipGAN inference
	return results 

def datagen(frames, mels, static):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if not static:
		face_det_results = face_detect([f[...,::-1] for f in frames]) # BGR2RGB for CNN face detection
	else:
		face_det_results = face_detect([frames[0][...,::-1]])

	for i, m in enumerate(mels):
		idx = 0 if static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords, valid_frame = face_det_results[idx].copy()
		if not valid_frame:
			print ("Face not detected, skipping frame {}".format(i))
			continue

		face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= lipgan_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, IMG_SIZE//2:] = 0

			img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, IMG_SIZE//2:] = 0

		img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 27
mel_idx_multiplier = 80./fps

def recognize(static):
	if static:
		full_frames = [cv2.imread(args.input[0])]
	else:
		video_stream = cv2.VideoCapture(args.video)
		
		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			full_frames.append(frame)
			if len(full_frames) % 2000 == 0: print(len(full_frames))

			if len(full_frames) * (1./fps) >= max_sec: break

		print ("Number of frames available for inference: "+str(len(full_frames)))

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan!')

	mel_chunks = []
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	batch_size = lipgan_batch_size
	gen = datagen(full_frames.copy(), mel_chunks, static)

	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(path.join(args.savepath, 'result.mp4'), 
									cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

		# expect
		# normal : bx96x96x6, bx12x35x1
		# mel : bx96x96x6, bx80x27x1
		#import time
		#start = int(round(time.time() * 1000))
		#print(img_batch.shape)
		#print(mel_batch.shape)
		pred = ailia_net.run([mel_batch, img_batch])[0]
		#end = int(round(time.time() * 1000))
		#print(f'\tailia processing time {end - start} ms')
		pred = pred * 255
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p, (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

	command = 'ffmpeg -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, path.join(args.savepath, 'result.mp4'), 
														path.join(args.savepath, 'result_voice.mp4'))
	subprocess.call(command, shell=True)

def main():
	# Check model files and download
	#check_and_download_models(
	#    WEIGHT_PATH,
	#    MODEL_PATH,
	#    REMOTE_PATH,
	#)
	#if not args.use_dlib:
	#    check_and_download_models(
	#        FACE_ALIGNMENT_WEIGHT_PATH,
	#        FACE_ALIGNMENT_MODEL_PATH,
	#        FACE_ALIGNMENT_REMOTE_PATH,
	#    )
	#    check_and_download_models(
	#        FACE_DETECTOR_WEIGHT_PATH,
	#        FACE_DETECTOR_MODEL_PATH,
	#        FACE_DETECTOR_REMOTE_PATH,
	#    )

	static = args.video is None

	if args.video is not None:
		# Video mode
		recognize(static)
	else:
		# Image mode
		recognize(static)


if __name__ == "__main__":
	main()