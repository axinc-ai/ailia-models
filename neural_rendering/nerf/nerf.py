import sys
import numpy as np
import imageio
import cv2

import utils_nerf
from load_llff import load_llff_data

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models


# logger
from logging import getLogger

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
#basedir = './nerf_llff_data'
config = 'config.txt'
IMAGE_PATH = './sample.png'
SAVE_IMAGE_PATH = './output.png'

H = 756.0
W = 1008.0
C = 3
focal = 815.1316

WEIGHT_PATH = "nerf.opt.onnx"
MODEL_PATH = "nerf.opt.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/nerf/"

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'run nerf',
    IMAGE_PATH,
    SAVE_IMAGE_PATH
)
parser = utils_nerf.config_parser(parser)
parser.add_argument("--config {}".format(config))
parser.add_argument(
    '-a', '--angle',
    default=0, type=int,
    help='Rendering angle (0 - 120)'
)
parser.add_argument(
    '-d', '--down_sample',
    default=8, type=int,
    help='Down sampling rate'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)

# ======================
# Main functions
# ======================
def main():
    # net initialize
    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    render_poses = load_llff_data(args.datadir, args.factor,
                                    recenter=True, bd_factor=.75,
                                    spherify=args.spherify,
                                    image_shape=(H, W, C))

    render_kwargs = utils_nerf.create_nerf(args, net)
    bds_dict = {
        'near': 0.,
        'far': 1.,
    }

    render_kwargs.update(bds_dict)
    render_kwargs_fast = {k : render_kwargs[k] for k in render_kwargs}
    render_kwargs_fast['N_importance'] = 0

    down = args.down_sample
    logger.info("Rendering angles "+str(len(render_poses)))
    logger.info("Rendering resolution "+str(int(W) // down)+"x"+str(int(H) // down))

    frames = []
    for i, c2w in enumerate(render_poses):
        if i!=args.angle:
            continue

        logger.info("Rendering angle "+str(i))
        test = utils_nerf.render(int(H) // down, int(W) // down, focal / down, c2w=c2w[:3, :4], **render_kwargs_fast)
        frame = (255 * np.clip(test[0], 0, 1)).astype(np.uint8)
        frames.append(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        logger.info(f'saved at : {args.savepath}')
        cv2.imwrite(args.savepath,frame)

    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')


if __name__=='__main__':
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    main()