import time
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable, Function

import ailia

# import original moduls
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
#WEIGHT_PATH = "wavenet_pytorch.onnx"
#MODEL_PATH = "wavenet_pytorch.onnx.prototxt"
#REMOTE_PATH = "../../../test/"
WEIGHT_PATH = "./wavenet_pytorch_op_11.onnx"
MODEL_PATH = "./wavenet_pytorch_op_11.onnx.prototxt"


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('pytorch_wavenet', './dammy.wav', './output.txt', input_ftype='audio')
parser.add_argument(
    '-n', '--onnx',
    action='store_true',
    default=False,
    help='Use onnxruntime'
)
args = update_parser(parser)


# ======================
# Main function
# ======================
def generate_first_sample():
    first_samples = torch.LongTensor(1).zero_() + (256 // 2)
    first_samples = Variable(first_samples)
    return first_samples


def get_model():
    if not args.onnx :
        logger.info('Use ailia')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else :
        logger.info('Use onnxruntime')
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)
    return net


def generate_wave(net, samples, num_generate=16000, temperature=1.0, regularize=0.):
    tic = time.time()

    num_samples = samples.size(0)
    input = Variable(torch.FloatTensor(1, 256, 1).zero_())
    input = input.scatter_(1, samples[0:1].view(1, -1, 1), 1.)

    print("fill queues with given samples")
    for i in range(num_samples - 1):
        x = _inference(net, input)
        input.zero_()
        input = input.scatter_(1, samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, 256, 1)

    print("generate new samples")
    generated = np.array([])
    regularizer = torch.pow(Variable(torch.arange(256)) - 256 / 2., 2)
    regularizer = regularizer.squeeze() * regularize

    for i in range(num_generate):
        output = _inference(net, input) # output[0].shape = (1, 256, 1)

        x = output.squeeze()

        x -= regularizer

        # sample from softmax distribution
        x /= temperature
        prob = F.softmax(x, dim=0)
        prob = prob.cpu()
        np_prob = prob.data.numpy()
        x = np.random.choice(256, p=np_prob)
        x = np.array([x])

        o = (x / 256) * 2. - 1
        generated = np.append(generated, o)

        # set new input
        x = Variable(torch.from_numpy(x).type(torch.LongTensor))
        input.zero_()
        input = input.scatter_(1, x.view(1, -1, 1), 1.).view(1, 256, 1)

    generated = (generated / 256) * 2. - 1
    mu_gen = _mu_law_expansion(generated, 256)

    toc = time.time()

    print("ailia processing does take {} seconds".format(str((toc-tic)*0.01)))
    return mu_gen


def output_wav(filename, rate, data):
    from scipy.io.wavfile import write
    write(filename, rate, data)


def _inference(net, input):
    input = input.to('cpu').detach().numpy().copy()
    if not args.onnx:
        output = net.predict(input)
    else:
        # TODO: fix original pytorch_wavenet code. there is missing input when exporting to onnx.
        output = net.run([net.get_outputs()[0].name], {})
        #output = net.run([net.get_outputs()[0].name], {"_": input})
        output = torch.from_numpy(output[0].astype(np.float32)).clone()
    return output


def _mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def main():
    # model files check and download
    #check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # create instance
    net = get_model()

    # generate sample
    sample = generate_first_sample()

    # generate wave
    generated = generate_wave(net, sample, num_generate=160000)

    # output wav
    output_wav('output.wav', 16000, generated)

    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
