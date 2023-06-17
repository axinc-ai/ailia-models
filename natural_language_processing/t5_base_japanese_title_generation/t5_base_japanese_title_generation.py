import argparse
import logging
import os
import re
import sys
import unicodedata

from transformers import T5Tokenizer
from onnxt5 import GenerativeT5
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402

# logger
logger = logging.getLogger(__name__)

"""
params
"""

INPUT_PATH = "./input.txt"
OUTPUT_PATH = "./output.txt"

HUGGING_FACE_MODEL_PATH = "sonoisa/t5-base-japanese-title-generation"
ENCODER_ONNX_PATH = "./t5-base-japanese-title-generation-encoder.onnx"
ENCODER_PROTOTXT_PATH = "./t5-base-japanese-title-generation-encoder.onnx.prototxt"
DECODER_ONNX_PATH = "./t5-base-japanese-title-generation-decoder-with-lm-head.onnx"
DECODER_PROTOTXT_PATH = "./t5-base-japanese-title-generation-decoder-with-lm-head.onnx.prototxt"

MAX_SOURCE_LENGTH = 512

"""
pre process functions
"""
def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

def normalize_text(text: str) -> str:
    assert "\n" not in text and "\r" not in text
    text = text.replace("\t", " ")
    text = text.strip()
    text = normalize_neologd(text)
    text = text.lower()
    return text

def preprocess_body(text: str) -> str:
    return normalize_text(text.replace("\n", " "))

"""
parse args
"""
parser = get_base_parser(
    description="T5 base Japanese title generation",
    default_input=INPUT_PATH,
    default_save=OUTPUT_PATH,
    input_ftype="text",
)
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
args = update_parser(parser)

def main(args):
    # download onnx and prototxt
    # TODO

    # load model
    tokenizer = T5Tokenizer.from_pretrained(HUGGING_FACE_MODEL_PATH, is_fast=True)
    if args.onnx:
        from onnxruntime import InferenceSession
        encoder_sess = InferenceSession(ENCODER_ONNX_PATH)
        decoder_sess = InferenceSession(DECODER_ONNX_PATH)
        model = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
    else:
        raise Exception("Only onnx runtime mode is supported. Please specify -o option to use onnx runtime.")
        # import ailia
        # import torch
        # encoder_sess = ailia.Net(ENCODER_PROTOTXT_PATH, ENCODER_ONNX_PATH)
        # decoder_sess = ailia.Net(DECODER_PROTOTXT_PATH, DECODER_ONNX_PATH)
        # generated = torch.tensor(tokenizer(prompt)['input_ids'])[:MAX_SOURCE_LENGTH - 1].unsqueeze(0)
        # model = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)

    for input_path in args.input:
        # load input file
        with open(input_path, "r") as fi:
            body = fi.read()

        # pre process
        body_preprocessed = preprocess_body(body)

        # execute prediction
        most_plausible_title, _ = model(body_preprocessed, 21, temperature=0.)
        logger.info("title: %s", most_plausible_title)
        save_path = get_savepath(args.savepath, input_path)
        with open(save_path, "a") as fo:
            fo.write(most_plausible_title)


if __name__ == '__main__':
    main(args)
