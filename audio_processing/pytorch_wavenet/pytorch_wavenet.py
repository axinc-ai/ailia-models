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
WEIGHT_PATH = "wavenet_pytorch_op_11.onnx"
MODEL_PATH = "wavenet_pytorch_op_11.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/pytorch-wavenet/"


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
def get_model():
    if not args.onnx :
        logger.info('Use ailia')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else :
        logger.info('Use onnxruntime')
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)
    return net


def generate_first_sample():
    first_samples = torch.randint(0, 256, (1028,))
    #first_samples = torch.full((1028,),fill_value=126).long() # silence wave sample
    return first_samples


def output_wav(filename, rate, data):
    from scipy.io.wavfile import write
    write(filename, rate, data)


def generate_wave(net, num_samples, first_samples=None, temperature=1.):
    tic = time.time()

    receptive_field = 1021
    classes = 256

    generated = Variable(first_samples, volatile=True)
    num_pad = receptive_field - generated.size(0)

    #if num_pad > 0:
    #    generated = _constant_pad_1d(generated, self.scope, pad_start=True)
    #    print("pad zero")

    for i in range(num_samples):
        input = Variable(torch.FloatTensor(1, classes, receptive_field).zero_())
        input = input.scatter_(1, generated[-receptive_field:].view(1, -1, receptive_field), 1.)

        output = _inference(net, input)

        x = output[:, :, -1].squeeze()

        if temperature > 0:
            x /= temperature
            prob = F.softmax(x, dim=0)
            prob = prob.cpu()
            np_prob = prob.data.numpy()
            x = np.random.choice(classes, p=np_prob)
            x = Variable(torch.LongTensor([x]))#np.array([x])
        else:
            x = torch.max(x, 0)[1].float()

        generated = torch.cat((generated, x), 0)

        # progress feedback
        if i % 1600 == 0:
            print(str(100 * i // num_samples) + "% generated")

    generated = (generated.float() / classes) * 2. - 1

    mu_gen = _mu_law_expansion(generated, classes)
    mu_gen = mu_gen.to('cpu').detach().numpy().copy() #to numpy

    toc = time.time()
    print("ailia processing does take {} seconds".format(str(toc-tic)))

    return mu_gen

    """
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
    """


def _inference(net, input):
    input = input.to('cpu').detach().numpy().copy()
    if not args.onnx:
        output = net.predict(input)
    else:
        output = net.run([net.get_outputs()[0].name], {net.get_inputs()[0].name: input})

    return torch.from_numpy(output[0].astype(np.float32)).clone()


def _mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def _constant_pad_1d(input, target_size, dimension=0, value=0, pad_start=False):
    num_pad = target_size - input.size(dimension)
    assert num_pad >= 0, 'target size has to be greater than input size'

    input_size = input.size()

    size = list(input.size())
    size[dimension] = target_size
    output = input.new(*tuple(size)).fill_(value)
    c_output = output

    # crop output
    if pad_start:
        c_output = c_output.narrow(dimension, num_pad, c_output.size(dimension) - num_pad)
    else:
        c_output = c_output.narrow(dimension, 0, c_output.size(dimension) - num_pad)

    c_output.copy_(input)
    return output


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # create instance
    net = get_model()

    # generate sample
    sample = generate_first_sample()

    # generate wave
    generated = generate_wave(net, 16000, first_samples=sample, temperature=1.0)

    # output wav
    output_wav('output.wav', 1600, generated)

    print(generated)

    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
