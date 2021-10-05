import sys, os
import time
import copy
import json
from logging import getLogger
import numpy as np
import cv2
import ailia
import glob
import torch
from tqdm import tqdm

sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

logger = getLogger(__name__)



# ======================
# Parameters
# ======================
WEIGHT_PATH = 'InvDN.onnx'
MODEL_PATH = 'InvDN.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/invertible_denoising_network/'
IMAGE_PATH = './input_images/'
SAVE_IMAGE_PATH = './output_images/'



# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Invertible Denoising Network', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-n', '--onnx',
    action='store_true',
    default=False,
    help='Use onnxruntime'
)
parser.add_argument(
    '-on', '--outname',
    default='sample',
)
args = update_parser(parser)



# ======================
# Main functions
# ======================
class Net():
    def __init__(self):
        if args.onnx:
            import onnxruntime
            self.net = onnxruntime.InferenceSession(WEIGHT_PATH)
        else:
            print('Waiting SDK update.')
            self.net = None #self.net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
            exit()

    def predict(self, input):
        if args.onnx:
            # preprocess
            input = input.astype(np.float32) / 255. # image array to Numpy float32, HWC, BGR, [0,1]
            input = np.expand_dims(input, 0)
            input = np.transpose(input, (0, 3, 1, 2))
            input = {
                self.net.get_inputs()[0].name: input.astype(np.float32),
                self.net.get_inputs()[1].name: np.array([1], dtype=np.int32)
            }
            # predict
            preds = self.net.run(None, input)
            # postprocess
            output = preds[1]
            output = output[:, :3, :, :]
            output = self.output2img_real(output)
            output = np.transpose(output, (1, 2, 0))
        else:
            #preds = self.net.predict({
            #    'input': input.astype(np.float32),
            #    'gaussian_scale': np.array([1])
            #})
            print('Waiting SDK update.')
            exit()
        return output

    def output2img_real(self, output, out_type=np.uint8, min_max=(0, 1)):
        darr = np.clip(np.squeeze(output), *min_max)
        darr = (darr - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
        if out_type == np.uint8:
            darr = (darr * 255.0).round()
        return darr.astype(out_type)

def add_noise(img, noise_param=50):
    height, width = img.shape[0], img.shape[1]
    std = np.random.uniform(0, noise_param)
    noise = np.random.normal(0, std, (height, width, 3))
    noise_img = np.array(img) + noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    return noise_img

def get_videoWriter(cap, name):
    return cv2.VideoWriter(
        name,
        cv2.VideoWriter_fourcc('m','p','4', 'v'), #mp4フォーマット
        cap.get(cv2.CAP_PROP_FPS), #fps
        (256, 256) #size
    )

def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH) # model files check and download

    net = Net()

    if args.ftype == 'image':
        print('ftype you choose is "image".')
        inputs = glob.glob(args.input[0]) if args.input else glob.glob("./input_images/*.PNG")
        for input in tqdm(inputs):
            basename = os.path.basename(input)
            cv2.imwrite(
                os.path.join('./output_images', basename),
                net.predict(cv2.imread(input))
            )

    elif args.ftype == 'video':
        print('ftype you choose is "video".')
        if args.video is None: # video file
            if not args.input:
                print('invalid video input')
                exit()
            else:
                print('processing from video file...')

            cap = cv2.VideoCapture(args.input[0])
            if not cap.isOpened():
                exit()

            video_name = os.path.splitext(os.path.basename(args.input[0]))[0]
            noised_video = get_videoWriter(cap, './output_videos/{}_noised.mp4'.format(video_name))
            denoised_video = get_videoWriter(cap, './output_videos/{}_denoised.mp4'.format(video_name))

            i = 1
            pbar = tqdm(total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            while True:
                pbar.update(i)
                ret, frame = cap.read()
                if not ret:
                    break

                noised_frame = add_noise(frame)

                noised_video.write(noised_frame)
                denoised_video.write(net.predict(noised_frame))

            cap.release()
            noised_video.release()
            denoised_video.release()

        else: # web camera
            print('processing from web camera...')

            cap = webcamera_utils.get_capture(args.video)
            if not cap.isOpened():
                exit()

            raw_video = get_videoWriter(cap, './output_videos/webcam_{}_raw.mp4'.format(args.outname))
            noised_video = get_videoWriter(cap, './output_videos/webcam_{}_noised.mp4'.format(args.outname))
            denoised_video = get_videoWriter(cap, './output_videos/webcam_{}_denoised.mp4'.format(args.outname))

            while(True):
                ret, frame = cap.read()
                if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
                    break

                _, resized_image = webcamera_utils.adjust_frame_size(frame, 256, 256)
                noised_frame = add_noise(resized_image)

                raw_video.write(resized_image)
                noised_video.write(noised_frame)
                denoised_video.write(net.predict(noised_frame))

            cap.release()
            raw_video.release()
            noised_video.release()
            denoised_video.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
