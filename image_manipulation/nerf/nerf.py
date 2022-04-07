import sys
import numpy as np
import imageio

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
basedir = './data/nerf_llff_data'
config = 'config.txt'
IMAGE_PATH = basedir + '/images_4/image000.png'
SAVE_IMAGE_PATH = basedir + '/output/sample.mp4'

H = 756.0
W = 1008.0
focal = 815.1316

WEIGHT_PATH = "nerf.opt.onnx"
MODEL_PATH = "nerf.opt.onnx.prototxt"

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
args = update_parser(parser)

# ======================
# Main functions
# ======================
def main():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    render_poses = load_llff_data(args.datadir, args.factor,
                                    recenter=True, bd_factor=.75,
                                    spherify=args.spherify)

    render_kwargs = utils_nerf.create_nerf(args, net)
    bds_dict = {
        'near': 0.,
        'far': 1.,
    }

    render_kwargs.update(bds_dict)
    render_kwargs_fast = {k : render_kwargs[k] for k in render_kwargs}
    render_kwargs_fast['N_importance'] = 0

    down = 8
    frames = []
    for i, c2w in enumerate(render_poses):
        if i % 8 == 0: print(i)
        test = utils_nerf.render(int(H) // down, int(W) // down, focal / down, c2w=c2w[:3, :4], **render_kwargs_fast)
        frames.append((255 * np.clip(test[0], 0, 1)).astype(np.uint8))

    logger.info(f'saved at : {args.savepath}')
    imageio.mimwrite(args.savepath, frames, fps=30, quality=8)
    logger.info('Script finished successfully.')


if __name__=='__main__':
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    main()