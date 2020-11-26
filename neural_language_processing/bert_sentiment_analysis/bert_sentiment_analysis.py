from transformers import DistilBertTokenizer

import numpy
import time
import sys
import argparse

import ailia

sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402

# ======================
# Arguemnt Parser Config
# ======================

DEFAULT_TEXT = 'Transformers and ailia SDK is an awesome combo!'
#DEFAULT_TEXT = "I'm sick today."

parser = argparse.ArgumentParser(
    description='bert sentiment-analysis.'
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

WEIGHT_PATH = "distilbert-base-uncased-finetuned-sst-2-english.onnx"
MODEL_PATH = "distilbert-base-uncased-finetuned-sst-2-english.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_sentiment_analysis/"


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    ailia_model = ailia.Net(MODEL_PATH,WEIGHT_PATH)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    model_inputs = tokenizer.encode_plus(args.input, return_tensors="pt")
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

    print("Input : ", args.input)

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

    score = numpy.exp(score) / numpy.exp(score).sum(-1, keepdims=True)

    label_name=["negative","positive"]

    label_id=numpy.argmax(numpy.array(score))
    print("Label : ",label_name[label_id])
    print("Score : ",score[0][0][label_id])

    print('Script finished successfully.')

if __name__ == "__main__":
    main()
