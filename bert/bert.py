import time
import os
import urllib.request

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer 

import ailia


weight_path = "bert-base-uncased.onnx"
model_path = "bert-base-uncased.onnx.prototxt"

rmt_ckpt = "https://storage.googleapis.com/ailia-models/bert_en/"

if not os.path.exists(model_path):
    urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)


def text2token(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # convert a text to tokens which can be interpreted in BERT model
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)

    tokens_ts = torch.tensor([indexed_tokens]).numpy()
    segments_ts = torch.tensor([segments_ids]).numpy()

    # length fixed by 128 for now (it can be changeable?)
    print(tokens_ts.shape)
    tokens_ts = np.pad(
        tokens_ts,
        [(0, 0), (0, 128-len(tokens_ts[0]))],
        'constant',
        # constant_values=(0, 0)
    )
    segments_ts = np.pad(
        segments_ts,
        [(0, 0), (0, 128-len(segments_ts[0]))],
        'constant',
        # constant_values=(0, 0)
    )
    return tokens_ts, segments_ts


def main():
    # load data
    # 1. DEBUG mode
    dummy_input = np.ones((1, 128))
    # dummy_inputs = np.array([dummy_input, dummy_input, dummy_input])

    # 2. input sample sentence
    # TODO how to implemenmt this ?
    test_sentence = 'Who was Jim Henson ? Jim Henson was a puppeteer.'
    tokens_ts, segments_ts = text2token(test_sentence)
    input_data = np.array([tokens_ts, segments_ts])
    print(f'[DEBUG] input shape: {input_data.shape}')

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(env_id)
    net = ailia.Net(model_path, weight_path, env_id=env_id)

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

    print(f'[DEBUG] output length: {len(preds_ailia)}')
    print(f'[DEBUG] output shape: {preds_ailia[0].shape}')
    print(f'[DEBUG] output shape: {preds_ailia[1].shape}')

    print('Successfully finished!')
    

if __name__ == "__main__":
    main()
