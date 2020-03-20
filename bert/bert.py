import time
import os
import urllib.request
import argparse

import numpy as np
from pyknp import Juman
from transformers import BertTokenizer

import ailia


# argument config
LANGS = ['en', 'jp']
parser = argparse.ArgumentParser()
parser.add_argument(
    '--lang', '-l', metavar='LANG', default='en', choices=LANGS,
    help='choose language: ' + ' | '.join(LANGS) + ' (default: en)'
)
args = parser.parse_args()
LANG = args.lang
print('[INFO] language is set to ' + LANG)


# Params
NUM_PREDICT = 10
if LANG == 'en':
    # masked word should be represented by '_'
    SENTENCE = 'I want to _ the car because it is cheap.'

    WEIGHT_PATH = "bert-base-uncased.onnx"
    MODEL_PATH = "bert-base-uncased.onnx.prototxt"
    RMT_CKPT = "https://storage.googleapis.com/ailia-models/bert_en/"

    MAX_SEQ_LEN = 128

elif LANG == 'jp':
    # masked word should be represented by '＿' (zen-kaku)
    SENTENCE = '私は車が安いので＿したい．'
    # SENTENCE = '私はクリスマスプレゼントには＿が欲しい！'

    # kyoto univ
    WEIGHT_PATH = 'kyoto-bert-jp.onnx'
    MODEL_PATH = 'kyoto-bert-jp.onnx.prototxt'

    RMT_CKPT = "https://storage.googleapis.com/ailia-models/bert_jp/"

    MAX_SEQ_LEN = 512

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(RMT_CKPT + MODEL_PATH, MODEL_PATH)
if not os.path.exists(WEIGHT_PATH):
    urllib.request.urlretrieve(RMT_CKPT + WEIGHT_PATH, WEIGHT_PATH)


def text2token(text, tokenizer, lang='en'):
    # convert a text to tokens which can be interpreted in BERT model
    if lang == 'en':
        text = text.replace('_', '[MASK]')
        masked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(masked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    elif lang == 'jp':
        jumanapp = Juman()
        juman_res = jumanapp.analysis(text)
        tokenized_text = [mrph.midasi for mrph in juman_res.mrph_list()]
        tokenized_text.insert(0, '[CLS]')
        tokenized_text.append('[SEP]')
        tokenized_text = [
            '[MASK]' if token == '＿' else token for token in tokenized_text
        ]
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    masked_index = tokenized_text.index('[MASK]')
    segments_ids = [0] * len(tokenized_text)
    tokens_ts = np.array([indexed_tokens])
    segments_ts = np.array([segments_ids])

    # input length fixed by max_seq_len
    # (ailia should manage adoptable input size)
    tokens_ts = np.pad(
        tokens_ts,
        [(0, 0), (0, MAX_SEQ_LEN-len(tokens_ts[0]))],
        'constant',
    )
    segments_ts = np.pad(
        segments_ts,
        [(0, 0), (0, MAX_SEQ_LEN-len(segments_ts[0]))],
        'constant',
    )
    assert tokens_ts.shape == (1, MAX_SEQ_LEN)
    assert segments_ts.shape == (1, MAX_SEQ_LEN)
    return tokens_ts, segments_ts, masked_index


def main():
    # bert tokenizer
    if LANG == 'en':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif LANG == 'jp':
        tokenizer = BertTokenizer(
            'vocab.txt',
            do_lower_case=False,
            do_basic_tokenize=False
        )

    # Prepare data
    dummy_input = np.ones((1, MAX_SEQ_LEN), dtype=np.int64)
    tokens_ts, segments_ts, masked_index = text2token(
        SENTENCE, tokenizer, lang=LANG
    )
    input_data = np.array([tokens_ts, segments_ts])

    # net initialize
    env_id = ailia.get_gpu_environment_id()

    # onnx debug
    # import onnxruntime
    # ss = onnxruntime.InferenceSession(WEIGHT_PATH)

    # input_name_0 = ss.get_inputs()[0].name
    # input_name_1 = ss.get_inputs()[1].name
    # input_name_2 = ss.get_inputs()[2].name
    # out = ss.get_outputs()[0].name
    # preds_ailia = ss.run([out], {
    #     input_name_0: tokens_ts.astype(np.int64),
    #     input_name_1: segments_ts.astype(np.int64),
    #     input_name_2: dummy_input
    # })

    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # compute time
    for i in range(1):
        start = int(round(time.time() * 1000))
        input_blobs = net.get_input_blob_list()
        for i, idx in enumerate(input_blobs):
            if i < len(input_data):
                net.set_input_blob_data(input_data[i], idx)
            else:
                net.set_input_blob_data(dummy_input, idx)
        net.update()
        preds_ailia = net.get_results()

        # preds_ailia = net.predict(dummy_input)[0]
        end = int(round(time.time() * 1000))
        print("ailia processing time {} ms".format(end-start))

    # Masked Word Prediction
    predicted_indices = np.argsort(
        preds_ailia[0][0][masked_index]
    )[-NUM_PREDICT:][::-1]

    print(predicted_indices)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices)

    print('Input sentence: ' + SENTENCE)
    print(f'predicted top {NUM_PREDICT} words: {predicted_tokens}')
    print('Successfully finished!')


if __name__ == "__main__":
    main()
