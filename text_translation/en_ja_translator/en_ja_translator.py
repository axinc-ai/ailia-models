import time
import sys
import re
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable, Function

import ailia

# import original moduls
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

from Lang import Lang

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
ENCODER_WEIGHT_PATH = "en-ja-translator_encoder_pytorch.onnx"
ENCODER_MODEL_PATH = "en-ja-translator_encoder_pytorch.onnx.prototxt"
DECODER_WEIGHT_PATH = "en-ja-translator_decoder_pytorch.onnx"
DECODER_MODEL_PATH = "en-ja-translator_decoder_pytorch.onnx.prototxt"
#REMOTE_PATH = "https://storage.googleapis.com/ailia-models/en-ja-translator/"
REMOTE_PATH = "./"

EN_LANG_PATH = "en.pkl"
JA_LANG_PATH = "ja.pkl"
SOS_token = 0
EOS_token = 1
use_cuda = torch.cuda.is_available()
max_length = 30


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('en_ja_translator', './input.txt', './output.txt', input_ftype='text')
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
def get_vocab():
    try:
        input_lang = pickle.load(open(EN_LANG_PATH, "rb"))
        output_lang = pickle.load(open(JA_LANG_PATH, "rb"))
        return input_lang, output_lang
    except FileNotFoundError:
        raise Exception("Could not find {} or {}.".format(EN_LANG_PATH, JA_LANG_PATH))


def get_model():
    if not args.onnx:
        logger.info('Use ailia')
        encoder = ailia.Net(ENCODER_MODEL_PATH, ENCODER_WEIGHT_PATH, env_id=args.env_id)
        decoder = ailia.Net(DECODER_MODEL_PATH, DECODER_WEIGHT_PATH, env_id=args.env_id)
    else:
        logger.info('Use onnxruntime')
        import onnxruntime
        encoder = onnxruntime.InferenceSession(ENCODER_WEIGHT_PATH)
        decoder = onnxruntime.InferenceSession(DECODER_WEIGHT_PATH)
    return encoder, decoder


def translate(input_lang, output_lang, sentence):
    input_variable = Variable(sentence2indexes(input_lang, sentence))
    input_length = input_variable.size()[0]

    encoder_hidden = init_hidden()
    encoder_outputs = init_outputs()

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder_inference(encoder, input_variable[i], encoder_hidden)
        encoder_outputs[i] = encoder_outputs[i] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    encoder_outputs = encoder_outputs.unsqueeze(0)

    decoded_words = []

    for i in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder_inference(decoder, decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni.item()])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words


def init_hidden():
    hidden = Variable(torch.zeros(2, 1, 300))
    if use_cuda:
        return hidden.cuda()
    else:
        return hidden


def init_outputs():
    encoder_outputs = Variable(torch.zeros(30, 600))
    if use_cuda:
        return encoder_outputs.cuda()
    else:
        return encoder_outputs


def encoder_inference(net, input, hidden):
    input = input.to('cpu').detach().numpy().copy()
    hidden = hidden.to('cpu').detach().numpy().copy()
    if not args.onnx:
        outputs = net.predict(input, hidden)
    else:
        outputs = net.run([
            net.get_outputs()[0].name,
            net.get_outputs()[1].name
        ], {
            net.get_inputs()[0].name: input,
            net.get_inputs()[1].name: hidden
        })
        output = torch.from_numpy(outputs[0].astype(np.float32)).clone()
        hidden = torch.from_numpy(outputs[1].astype(np.float32)).clone()
    return output, hidden


def decoder_inference(net, input, hidden, encoder_outputs):
    input = input.to('cpu').detach().numpy().copy()
    hidden = hidden.to('cpu').detach().numpy().copy()
    encoder_outputs = encoder_outputs.to('cpu').detach().numpy().copy()
    if not args.onnx:
        outputs = net.predict(input, hidden, encoder_outputs)
    else:
        outputs = net.run([
            net.get_outputs()[0].name,
            net.get_outputs()[1].name,
            net.get_outputs()[2].name
        ], {
            net.get_inputs()[0].name: input,
            net.get_inputs()[1].name: hidden,
            net.get_inputs()[2].name: encoder_outputs
        })
        output = torch.from_numpy(outputs[0].astype(np.float32)).clone()
        hidden = torch.from_numpy(outputs[1].astype(np.float32)).clone()
        attention = torch.from_numpy(outputs[2].astype(np.float32)).clone()
    return output, hidden, attention


def normalize_en(s):
    """ Processes an English string by removing non-alphabetical characters (besides .!?).
    """
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^\w.!?]+", r" ", s, flags=re.UNICODE)
    return s


def sentence2indexes(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(" ")]
    result = torch.LongTensor(max_length)
    result[:] = EOS_token
    for i, index in enumerate(indexes):
        result[i] = index

    if use_cuda:
        return result.cuda()
    else:
        return result


def main():
    sentence = normalize_en('I am japanese.')
    
    # model files check and download
    #check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # create vocabulary
    input_lang, output_lang = get_vocab()

    # create instance
    encoder, decoder = get_model()

    # translate
    decoded_words = translate(sentence, input_lang, output_lang)

    print("".join(decoded_words[:-1]))

    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
