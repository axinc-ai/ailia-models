import sys

import os
import time

import cv2
import numpy as np
from PIL import Image

import ailia
from dataloaders.dataloader import MyDataloader
from dataloaders.nyu import NYUDataset
from metrics import AverageMeter, Result
import utils_misc

# Import original modules.
sys.path.append("../../util")
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # NOQA: E402
from webcamera_utils import get_capture, cut_max_square  # NOQA: E402

# Logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = "fast-depth.onnx"
MODEL_PATH = "fast-depth.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/fast-depth/"
DATA_NAMES = ["nyudepthv2"]
MODALITY_NAMES = MyDataloader.modality_names
PRINT_FREQ = 1
SOURCE_IMAGE_PATH = "data"
SAVE_IMAGE_PATH = "img"
IMAGE_PATH = "data/img/00001.png"
DEPTH_MIN = 0  # In meters.
DEPTH_MAX = 5  # In meters.

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser("FastDepth", SOURCE_IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "--data",
    metavar="DATA",
    default="nyudepthv2",
    choices=DATA_NAMES,
    help="dataset: " + " | ".join(DATA_NAMES) + " (default: nyudepthv2)",
)
parser.add_argument(
    "--validation_mode",
    action="store_true",
    help="Validation mode",
)
parser.add_argument(
    "--modality",
    "-m",
    metavar="MODALITY",
    default="rgb",
    choices=MODALITY_NAMES,
    help="Modality: " + " | ".join(MODALITY_NAMES) + " (default: rgb)",
)
parser.add_argument(
    "--savepath",
    default="img",
    type=str,
    metavar="PATH",
    help="Path to output directory",
)

args = update_parser(parser)


# ======================
# Main functions
# ======================
def _make_dataset(img, transformer):
    input_np, _ = transformer.val_transform(img, None)
    input_tensor = input_np.transpose((2, 0, 1)).copy()
    while input_tensor.ndim < 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return [np.expand_dims(input_tensor, 0)]


def _prepare_data(args, frame=None, transformer=None):
    if args.video is not None:
        return _make_dataset(frame, transformer)
    else:
        if args.validation_mode:
            # Data loading code
            logger.info("=> creating data loaders...")

            if args.data == "nyudepthv2":
                dat_dir = os.path.join(".", "data/h5", args.data, "val")
                dataset = NYUDataset(dat_dir, split="val", modality=args.modality)
                logger.info("=> data loaders created.")
                return dataset
            else:
                raise RuntimeError("Dataset not found.")
        else:
            path = os.path.join(".", IMAGE_PATH)
            with Image.open(path) as im:
                rgb = np.asarray(im)
            transformer = NYUDataset("./data", split="val", modality=args.modality)
            return _make_dataset(rgb, transformer)


def _initialize_net(args):
    return ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)


def _infer(img, net):
    return net.predict(img)


def _estimate(img, model):
    img_depth = _infer(img, model)
    depth_pred = np.squeeze(img_depth)
    depth_pred_col = utils_misc.colored_depthmap(
        depth_pred,
        min(DEPTH_MIN, np.min(depth_pred)),
        max(DEPTH_MAX, np.max(depth_pred)),
    )

    return depth_pred_col


def _validate(val_loader, model, args):
    average_meter = AverageMeter()
    end = time.time()
    for i, (img, target) in enumerate(val_loader):
        data_time = time.time() - end

        # Compute output.
        end = time.time()
        pred = _infer(img, model)
        cpu_time = time.time() - end

        # Measure accuracy and record loss.
        result = Result()
        result.evaluate(pred, np.asarray(target.data))
        average_meter.update(result, cpu_time, data_time, img.shape[0])
        end = time.time()

        if args.modality == "rgb":
            rgb = img

        if i == 0:
            img_merge = utils_misc.merge_into_row(rgb, target, pred)
        elif i < 8:
            row = utils_misc.merge_into_row(rgb, target, pred)
            img_merge = utils_misc.add_row(img_merge, row)
        elif i == 8:
            filename = args.savepath + "/comparison_" + args.data + ".png"
            utils_misc.save_image(img_merge, filename)

        if (i + 1) % PRINT_FREQ == 0:
            logger.info(
                "Test: [{0}/{1}]\t"
                "t_CPU={cpu_time:.3f}({average.gpu_time:.3f})\n\t"
                "RMSE={result.rmse:.2f}({average.rmse:.2f}) "
                "MAE={result.mae:.2f}({average.mae:.2f}) "
                "Delta1={result.delta1:.3f}({average.delta1:.3f}) "
                "REL={result.absrel:.3f}({average.absrel:.3f}) "
                "Lg10={result.lg10:.3f}({average.lg10:.3f}) ".format(
                    i + 1,
                    len(val_loader),
                    cpu_time=cpu_time,
                    result=result,
                    average=average_meter.average(),
                )
            )

    avg = average_meter.average()

    logger.info(
        "\n*\n"
        "RMSE={average.rmse:.3f}\n"
        "MAE={average.mae:.3f}\n"
        "Delta1={average.delta1:.3f}\n"
        "REL={average.absrel:.3f}\n"
        "Lg10={average.lg10:.3f}\n"
        "t_CPU={time:.3f}\n".format(average=avg, time=avg.cpu_time)
    )

    return avg, img_merge


def transfer_to_image():
    # Prepare input data.
    if args.validation_mode:
        logger.info("Validation mode")
    dataset = _prepare_data(args)

    # Initialize net.
    net = _initialize_net(args)

    # Inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        for i in range(5):
            start = int(round(time.time() * 1000))
            if args.validation_mode:
                _validate(dataset, net, args)
            else:
                _estimate(dataset[0], net)
            end = int(round(time.time() * 1000))
            logger.info(f"\tailia processing time {end - start} ms")
    else:
        if args.validation_mode:
            _validate(dataset, net, args)
        else:
            depth_pred_col = _estimate(dataset[0], net)
            stem = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
            filepath = os.path.join(args.savepath, f"{stem}_depth.png")
            utils_misc.save_image(depth_pred_col, filepath)
    logger.info("Script finished successfully.")


def transfer_to_video():
    # Initialize net.
    net = _initialize_net(args)

    # Initialize transformer.
    transformer = NYUDataset("./data", split="val", modality=args.modality)

    capture = get_capture(args.video)

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break

        # Prepare input data.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cut_max_square(frame)
        dataset = _prepare_data(args, frame, transformer)

        # Inference
        depth_pred_col = _estimate(dataset[0], net)

        # Postprocessing
        cv2.imshow(
            "frame", cv2.cvtColor(depth_pred_col.astype("uint8"), cv2.COLOR_RGB2BGR)
        )

    capture.release()
    cv2.destroyAllWindows()
    logger.info("Script finished successfully.")


def main():
    # Check model files and download.
    check_and_download_models(
        WEIGHT_PATH,
        MODEL_PATH,
        REMOTE_PATH,
    )

    if args.video is not None:
        # Video mode
        transfer_to_video()
    else:
        # Image mode
        transfer_to_image()


if __name__ == "__main__":
    main()
