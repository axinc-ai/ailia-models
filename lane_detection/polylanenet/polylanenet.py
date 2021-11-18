import os, sys
import glob
import time
import numpy as np
import cv2
from tqdm import tqdm
import ailia
# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)
# for PolyLaneNet
import torch
import torch.nn as nn


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'polylanenet.onnx'
MODEL_PATH = 'polylanenet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/polylanenet/'

IMAGE_PATH = 'input'
SAVE_IMAGE_PATH = 'output'

HEIGHT = 360
WIDTH = 640


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'PolyLaneNet', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '--input_type', choices=['image', 'video'], required=True
)
parser.add_argument(
    '--input_name', required=True
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def decode(output, conf_threshold=0.5, share_top_y=True):
    sigmoid = nn.Sigmoid()
    output = output.reshape(len(output), -1, 7)  # score + upper + lower + 4 coeffs = 7
    output[:, :, 0] = sigmoid(output[:, :, 0])
    output[output[:, :, 0] < conf_threshold] = 0

    if False and share_top_y:
        output[:, :, 0] = output[:, 0, 0].expand(output.shape[0], output.shape[1])

    return output


def draw_annotation(img, pred):
    img = (img/255.).astype(np.float32)
    # Unnormalize
    if False:
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
        IMAGENET_STD = np.array([0.229, 0.224, 0.225])
        img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = (img * 255).astype(np.uint8)

    img_h, img_w, _ = img.shape
    original = img.copy()

    # Draw predictions
    pred = pred[0]
    pred = pred[pred[:, 0] != 0]  # filter invalid lanes
    overlay = img.copy()
    for i, lane in enumerate(pred):
        lane = lane[1:]  # remove conf
        lower, upper = lane[0], lane[1]
        lane = lane[2:]  # remove upper, lower positions

        # generate points from the polynomial
        ys = np.linspace(lower, upper, num=100)
        points = np.zeros((len(ys), 2), dtype=np.int32)
        points[:, 1] = (ys * img_h).astype(int)
        points[:, 0] = (np.polyval(lane, ys) * img_w).astype(int)
        points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

        # draw lane with a polyline on the overlay
        for current_point, next_point in zip(points[:-1], points[1:]):
            overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=(0, 255, 0), thickness=2)

        # draw lane ID
        if len(points) > 0:
            cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0))

    predicted = img.copy()

    # Add lanes overlay
    w = 0.6
    img = ((1. - w) * img + w * overlay).astype(np.uint8)

    overlayed = img.copy()

    return original, predicted, overlayed


def get_files(dirname):
    files = glob.glob(dirname)
    if len(files)==0:
        print('specified files is empty.')
        exit()
    return files


def predict(net, input):
    # resize input image
    height = input.shape[0]
    width = input.shape[1]
    rate = HEIGHT/height
    rs_height = int(height*rate)
    rs_width = int(width*rate)
    if (rs_height!=HEIGHT or rs_width!=WIDTH):
        print('Invalid image shape')
        exit()
    input = cv2.resize(input, (rs_width, rs_height))
    image = input
    input = (input/255.).astype(np.float32)
    input = input.transpose(2, 0, 1)
    input = input[np.newaxis, :, :, :]
    # predict
    output = net.predict(input)
    # postprocess
    output = torch.from_numpy(output)
    output = decode(output)
    output = output.cpu().numpy()

    return output, image


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    if not os.path.exists('./output'):
        os.mkdir('./output')

    if args.ftype == 'image':
        if not os.path.exists('./output/image'):
            os.mkdir('./output/image')

        # image to image
        if args.input_type == 'image':
            files = get_files('./input/image/{}'.format(args.input_name))
            input = cv2.imread('{}'.format(files[0]))
            output, image = predict(net, input)
            _, _, overlayed = draw_annotation(image, output)
            cv2.imwrite('./output/image/{}'.format(args.input_name), overlayed)

        # video to image
        elif args.input_type == 'video':
            print('Not implemented.')
            exit()

    elif args.ftype == 'video':
        if not os.path.exists('./output/video'):
            os.mkdir('./output/video')

        # image to video
        if args.input_type == 'image':
            files = get_files('./input/video/{}/*.jpg'.format(args.input_name))
            filename = '_'.join(args.input_name.split('/'))+'.mp4'

            # sort
            tmp = [file.split('/')[-1].replace('.jpg', '') for file in files]
            tmp = sorted(tmp, key=int)
            files = ['./input/video/{}/{}.jpg'.format(args.input_name, file) for file in tmp]

            # create video
            video = cv2.VideoWriter(
                    './output/video/{}'.format(filename),
                    cv2.VideoWriter_fourcc('m','p','4', 'v'), #mp4フォーマット
                    float(30), #fps
                    (WIDTH, HEIGHT) #size
            )

            # pred
            for file in tqdm(files):
                input = cv2.imread('{}'.format(file))
                output, image = predict(net, input)
                _, _, overlayed = draw_annotation(image, output)
                video.write(overlayed)
            video.release()

        # video to video
        elif args.input_type == 'video':
            cap = cv2.VideoCapture('./input/video/{}'.format(args.input_name))
            if not cap.isOpened():
                print('Video is not opened.')
                exit()
            filename = '_'.join(args.input_name.split('/'))

            # create video
            output_video = cv2.VideoWriter(
                    './output/video/{}'.format(filename),
                    cv2.VideoWriter_fourcc('m','p','4', 'v'), #mp4フォーマット
                    float(30), #fps
                    (WIDTH, HEIGHT) #size
            )

            i = 1
            pbar = tqdm(total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            while True:
                pbar.update(i)
                ret, frame = cap.read()
                if not ret:
                    break
                output, image = predict(net, frame)
                _, _, overlayed = draw_annotation(image, output)
                output_video.write(overlayed)
            cap.release()
            output_video.release()

        print('Video saved as {}'.format(filename))

    else:
        print('invalid ftype.')
        exit()


if __name__ == '__main__':
    main()
