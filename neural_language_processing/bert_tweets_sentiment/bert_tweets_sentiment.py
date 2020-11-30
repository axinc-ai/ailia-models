from transformers import BertJapaneseTokenizer, BertForMaskedLM

import numpy
import time
import sys
import argparse

import ailia

sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402

# require ailia SDK 1.2.5 and later

# ======================
# Arguemnt Parser Config
# ======================

DEFAULT_TEXT = 'iPhone 12 mini が欲しい'
#DEFAULT_TEXT = 'iPhone 12 mini は高い'

parser = argparse.ArgumentParser(
    description='bert tweets sentiment.'
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
args = parser.parse_args()


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = "bert_tweets_sentiment.onnx"
MODEL_PATH = "bert_tweets_sentiment.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_tweets_sentiment/"


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    ailia_model = ailia.Net(MODEL_PATH,WEIGHT_PATH)
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    model_inputs = tokenizer.encode_plus(args.input, return_tensors="pt")
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

    print("Text : ", args.input)
    print("Input : ", inputs_onnx)

    # inference
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            score = ailia_model.predict(inputs_onnx)
            end = int(round(time.time() * 1000))
            print("\tailia processing time {} ms".format(end - start))
    else:
        score = ailia_model.predict(inputs_onnx)

    print("Output : ", score)

    label_name=["positive","negative"]

    print("Label : ",label_name[numpy.argmax(numpy.array(score))])

    print('Script finished successfully.')

if __name__ == "__main__":
    main()
