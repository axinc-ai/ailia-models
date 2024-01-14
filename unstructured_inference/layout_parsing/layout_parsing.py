import sys

import ailia

sys.path.append('../../util')
from logging import getLogger  # noqa: E402

from arg_utils import (get_base_parser, get_savepath,  # noqa: E402
                       update_parser)
from layout_parsing_utils import pdf_to_images
from model_utils import check_and_download_models  # noqa: E402


def infer():
    pass


if __name__ == "__main__":
    weight_path, model_path, params = get_params(args.arch, args.n_neighbors)
    check_and_download_models(weight_path, model_path, REMOTE_PATH)
    net = ailia.Net(model_path, weight_path, env_id=args.env_id)
    infer()
