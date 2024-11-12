import os
import sys
import ailia
import numpy as np
import json

# logger
from logging import getLogger  # noqa: E402

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

import pandas as pd
import matplotlib.pyplot as plt

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "timesfm-1.0-200m.onnx"
MODEL_PATH = "timesfm-1.0-200m.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/timesfm/"

DATA_PATH = "ETTh1.csv"
SAVE_DATA_PATH = "output.npy"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("TimesFM", DATA_PATH, SAVE_DATA_PATH)
parser.add_argument("-i", "--input", type=str, default=DATA_PATH)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime",
)
parser.add_argument(
    "-w",
    "--write_json",
    action="store_true",
    help="Flag to output results to json file.",
)
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def draw_result(history, trues, preds, save_path):
    plt.figure(figsize=(12, 4))

    # Plotting the first time series from history
    plt.plot(
        range(len(history)),
        history,
        label=f"History ({len(history)} timesteps)",
        c="darkblue",
    )

    offset = len(history)
    plt.plot(
        range(offset, offset + len(trues)),
        trues,
        label=f"Ground Truth ({len(trues)} timesteps)",
        color="darkblue",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        range(offset, offset + len(preds)),
        preds,
        label=f"Forecast ({len(preds)} timesteps)",
        color="red",
        linestyle="--",
    )

    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=14)

    plt.savefig(save_path)


# ======================
# Main functions
# ======================


def get_data(
    data_path,
):
    df = pd.read_csv(data_path)
    df = df[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]]
    return df


def time_series_forecasting(net):
    ### prepare dataset ###
    data_path = args.input[0]
    context_length = 512
    forecast_horizon = 96

    df_data = get_data(data_path)
    df_input = df_data.iloc[-(context_length + forecast_horizon) : -forecast_horizon]
    df_true = df_data.iloc[-forecast_horizon:]
    target_index = 6

    forecast_input = df_input.values.T
    point_true = df_true.values.T

    ### visualize ###
    draw_result(history, trues, preds, "output.png")

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # net initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    # forecasting
    time_series_forecasting(net)


if __name__ == "__main__":
    main()
