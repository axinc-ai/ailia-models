import time
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import soundfile as sf

import ailia

# import original moduls
sys.path.append("../../util")
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = "wavenet_pytorch_op_17.onnx"
MODEL_PATH = "wavenet_pytorch_op_17.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/pytorch-wavenet/"


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    "pytorch_wavenet", "./dammy.wav", "./output.txt", input_ftype="audio"
)
parser.add_argument(
    "-n", "--onnx", action="store_true", default=False, help="Use onnxruntime"
)
args = update_parser(parser)


# ======================
# Main function
# ======================
def get_model():
    if not args.onnx:
        logger.info("Use ailia")
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else:
        logger.info("Use onnxruntime")
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGHT_PATH)
    return net


def generate_first_sample():
    first_samples = torch.randint(0, 256, (3085,))
    return first_samples


def output_wav(filename, rate, data):
    sf.write(filename, rate, data)


def generate_wave(net, num_samples, first_samples=None, temperature=1.0):
    tic = time.time()

    progress_interval = 1000

    # receptive_field = 1021
    classes = 256

    if first_samples is None:
        first_samples = generate_first_sample()

    num_given_samples = first_samples.shape[0]
    total_samples = num_given_samples + num_samples

    input = np.zeros((1, classes, 1), dtype=np.float32)
    input[:, first_samples[0], :] = 1

    # prepare queues
    shapes = [
        # fmt: off
        (32, 2), (32, 3), (32, 5), (32, 9), (32, 17),
        (32, 33), (32, 65), (32, 129), (32, 257), (32, 513),
        (32, 2), (32, 3), (32, 5), (32, 9), (32, 17),
        (32, 33), (32, 65), (32, 129), (32, 257), (32, 513),
        (32, 2), (32, 3), (32, 5), (32, 9), (32, 17),
        (32, 33), (32, 65), (32, 129), (32, 257), (32, 513),
        # fmt: on
    ]
    dilated_queues = [np.zeros(shape, dtype=np.float32) for shape in shapes]

    # fill queues with given samples
    for i in range(num_given_samples - 1):
        _, dilated_queues = _inference(net, input, dilated_queues)

        input.fill(0)
        input[:, first_samples[i + 1], :] = 1

        if i % progress_interval == 0:
            print(str(100 * i // total_samples) + "% generated")

    # generated = Variable(first_samples, volatile=True)
    # num_pad = receptive_field - generated.size(0)

    # generate new samples
    for i in range(num_samples):
        x, dilated_queues = _inference(net, input, dilated_queues)
        x = x.squeeze()

        if temperature > 0:
            x /= temperature
            prob = F.softmax(x, dim=0)
            prob = prob.cpu()
            np_prob = prob.data.numpy()
            x = np.random.choice(classes, p=np_prob)
            x = Variable(torch.LongTensor([x]))  # np.array([x])
        else:
            x = torch.max(x, 0)[1].float()

        generated = torch.cat((generated, x), 0)

        # progress feedback
        if i % progress_interval == 0:
            print(str(100 * (i + num_given_samples) // total_samples) + "% generated")

    generated = (generated.float() / classes) * 2.0 - 1

    mu_gen = _mu_law_expansion(generated, classes)
    mu_gen = mu_gen.to("cpu").detach().numpy().copy()  # to numpy

    toc = time.time()
    print("ailia processing does take {} seconds".format(str(toc - tic)))

    return mu_gen


def _inference(net, input, dilated_queues):
    if not args.onnx:
        output = net.run([input, *dilated_queues])
    else:
        output = net.run(
            None,
            {
                "input": input,
                **{
                    "dilated_queues_%d" % (i + 1): q
                    for i, q in enumerate(dilated_queues)
                },
            },
        )
    x = output[0]
    dilated_queues = output[1:]

    return x, dilated_queues


def _mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # create instance
    net = get_model()

    # generate sample
    sample = generate_first_sample()

    # generate wave
    generated = generate_wave(net, 160000, first_samples=sample, temperature=1.0)

    # output wav
    output_wav("output.wav", 16000, generated)

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
