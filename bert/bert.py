import time
import os
import urllib.request

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer 

import ailia


NUM_PREDICT = 3

WEIGHT_PATH = "bert-base-uncased.onnx"
MODEL_PATH = "bert-base-uncased.onnx.prototxt"

RMT_CKPT = "https://storage.googleapis.com/ailia-models/bert_en/"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(RMT_CKPT + MODEL_PATH, MODEL_PATH)
if not os.path.exists(WEIGHT_PATH):
    urllib.request.urlretrieve(RMT_CKPT + WEIGHT_PATH, WEIGHT_PATH)


def text2token(text, tokenizer):
    # convert a text to tokens which can be interpreted in BERT model
    text = text.replace('_', '[MASK]')
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    masked_index = tokenized_text.index('[MASK]')
    segments_ids = [0] * len(tokenized_text)

    tokens_ts = torch.tensor([indexed_tokens]).numpy()
    segments_ts = torch.tensor([segments_ids]).numpy()

    # length fixed by 128 for now (it can be changeable?)
    tokens_ts = np.pad(
        tokens_ts,
        [(0, 0), (0, 128-len(tokens_ts[0]))],
        'constant',
    )
    segments_ts = np.pad(
        segments_ts,
        [(0, 0), (0, 128-len(segments_ts[0]))],
        'constant',
    )
    assert tokens_ts.shape == (1, 128)
    assert segments_ts.shape == (1, 128)
    return tokens_ts, segments_ts, masked_index


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Prepare data
    dummy_input = np.ones((1, 128))
    test_sentence = 'I want to _ the car because it is cheap.'
    tokens_ts, segments_ts, masked_index = text2token(test_sentence, tokenizer)
    input_data = np.array([tokens_ts, segments_ts])
    
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # compute time
    for i in range(5):
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
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices)
    print(f'predicted top {NUM_PREDICT} words: {predicted_tokens}')
    
    print('Successfully finished!')
    

if __name__ == "__main__":
    main()
