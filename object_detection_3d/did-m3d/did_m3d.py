import sys
import cv2
import yaml
import time
import ailia
import numpy as np
from did_m3d_util import Tester, get_objects_from_label

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
MODEL_NAME = 'did_m3d'
WEIGHT_PATH = MODEL_NAME + '.onnx'
MODEL_PATH  = MODEL_NAME + '.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/did_m3d/'

IMAGE_PATH = '000005.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Depth estimation model', IMAGE_PATH, SAVE_IMAGE_PATH)

parser.add_argument('--config', type=str, default='kitti.yaml')
parser.add_argument('--calib_path', type=str, default='000005.txt')

args = update_parser(parser)

def visualization(savepath):
    objects = get_objects_from_label("output_tmp.txt")

    with open(args.calib_path, 'r') as f:
        lines = f.readlines()
        P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

    img = cv2.imread(args.input[0])

    def line(img,p1, p2):
        return cv2.line(img,(int(p1[0]), int(p1[1])),
                            (int(p2[0]), int(p2[1])),
                             color=(0,255,0))

    for i in range(len(objects)):
        corners_3d = objects[i].generate_corners3d()
        corners_3d_hom = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
        corners_img = np.matmul(corners_3d_hom , P2.T)
        corners_img = corners_img[:, :2] / corners_img[:, 2][:, None]

        # draw the upper 4 horizontal lines
        img = line(img,corners_img[0], corners_img[1])
        img = line(img,corners_img[1], corners_img[2])
        img = line(img,corners_img[2], corners_img[3])
        img = line(img,corners_img[3], corners_img[0])

        #dras,w the lower 4 horizontal lines
        img = line(img,corners_img[4], corners_img[5])
        img = line(img,corners_img[5], corners_img[6])
        img = line(img,corners_img[6], corners_img[7])
        img = line(img,corners_img[7], corners_img[4])

        #dras,w the 4 vertical lines
        img = line(img,corners_img[4], corners_img[0])
        img = line(img,corners_img[5], corners_img[1])
        img = line(img,corners_img[6], corners_img[2])
        img = line(img,corners_img[7], corners_img[3])
    cv2.imwrite(savepath,img)
    


# ======================
# Main functions
# ======================
def didm3d_from_image():
    # net initialize
    net = ailia.Net(None,"did_m3d.onnx")
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    tester = Tester(net,cfg['dataset'],th=0.3)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = tester.test(image_path,args.calib_path)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = tester.test(image_path,args.calib_path)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        visualization(savepath)
        logger.info(f'saved at : {savepath}')
    logger.info('Script finished successfully.')

def main():
    # model files check and download
    #check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # image mode
    didm3d_from_image()

if __name__ == '__main__':
    main()
