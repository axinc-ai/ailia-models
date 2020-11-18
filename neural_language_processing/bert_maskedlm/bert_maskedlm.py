import sys
import torch
import numpy
import argparse

from transformers import BertTokenizer, BertJapaneseTokenizer, BertForMaskedLM

import ailia

sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402


# ======================
# Arguemnt Parser Config
# ======================

MODEL_LISTS = [
    'bert-base-cased',
    'bert-base-uncased',
    'bert-base-japanese-whole-word-masking'
]

DEFAULT_TEXT = '私はお金で動く。'

parser = argparse.ArgumentParser(
    description='bert masklm sample.'
)

parser.add_argument(
    '--input', '-i', metavar='TEXT',
    default=DEFAULT_TEXT, 
    help='input text'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='bert-base-japanese-whole-word-masking', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
args = parser.parse_args()


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = args.arch+".onnx"
MODEL_PATH = args.arch+".onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_maskedlm/"


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.arch=='bert-base-cased' or args.arch=='bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(args.arch)
    else:
        tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/'+'bert-base-japanese-whole-word-masking')
    text = args.input
    print("Input text : "+text)

    tokenized_text = tokenizer.tokenize(text)
    print("Tokenized text : ",tokenized_text)

    masked_index = 2
    tokenized_text[masked_index] = '[MASK]'
    print("Masked text : ",tokenized_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    print("Indexed tokens : ",indexed_tokens)

    ailia_model = ailia.Net(MODEL_PATH,WEIGHT_PATH)

    indexed_tokens = numpy.array(indexed_tokens)
    token_type_ids = numpy.zeros((1,len(tokenized_text)))
    attention_mask = numpy.zeros((1,len(tokenized_text)))

    inputs_onnx = {"token_type_ids":token_type_ids,"input_ids":indexed_tokens,"attention_mask":attention_mask}

    print("Predicting...")
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            outputs = ailia_model.predict(inputs_onnx)
            end = int(round(time.time() * 1000))
            print("\tailia processing time {} ms".format(end - start))
    else:
        outputs = ailia_model.predict(inputs_onnx)

    print("Output : ",outputs)

    predictions = torch.from_numpy(outputs[0][0, masked_index]).topk(5)

    print("Predictions : ")
    for i, index_t in enumerate(predictions.indices):
        index = index_t.item()
        token = tokenizer.convert_ids_to_tokens([index])[0]
        print(i, token)

    print('Script finished successfully.')

if __name__ == "__main__":
    main()
