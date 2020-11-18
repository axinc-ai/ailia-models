from transformers import AutoTokenizer

import numpy
import time
import sys
import argparse

import ailia

#sys.path.append('../../util')
#from model_utils import check_and_download_models  # noqa: E402

#O : 固有表現外
#B-MIS : 別のその他の直後のその他の始まり
#I-MIS : その他
#B-PER : 別の人物名の直後の人物名の始まり
#I-PER : 人物名
#B-ORG : 別の組織の直後の組織の始まり
#I-ORG : 組織
#B-LOC : 別の場所の直後の場所の始まり
#I-LOC : 場所

# ======================
# Arguemnt Parser Config
# ======================

DEFAULT_TEXT = 'My name is bert'

parser = argparse.ArgumentParser(
    description='bert ner.'
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

WEIGHT_PATH = "../onnx_transformers/.onnx/dbmdz/bert-large-cased-finetuned-conll03-english/bert-large-cased-finetuned-conll03-english.onnx"
MODEL_PATH = "../onnx_transformers/.onnx/dbmdz/bert-large-cased-finetuned-conll03-english/bert-large-cased-finetuned-conll03-english.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_tweets_sentiment/"


# ======================
# Main function
# ======================
def main():
    # model files check and download
    #check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    ailia_model = ailia.Net(MODEL_PATH,WEIGHT_PATH)
    tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
    model_inputs = tokenizer.encode_plus(args.input, return_tensors="pt")
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

    print("Input : ", args.input)

    # inference
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            entities = ailia_model.predict(inputs_onnx)
            end = int(round(time.time() * 1000))
            print("\tailia processing time {} ms".format(end - start))
    else:
        entities = ailia_model.predict(inputs_onnx)

    id2label = {0: 'O', 1: 'B-MISC', 2: 'I-MISC', 3: 'B-PER', 4: 'I-PER', 5: 'B-ORG', 6: 'I-ORG', 7: 'B-LOC', 8: 'I-LOC'}
    ignore_labels = ['O']

    score = numpy.exp(entities) / numpy.exp(entities).sum(-1, keepdims=True)
    labels_idx = score.argmax(axis=-1)
    labels_idx = labels_idx[0][0]

    entities = []

    filtered_labels_idx = [
        (idx, label_idx)
        for idx, label_idx in enumerate(labels_idx)
        if id2label[label_idx] not in ignore_labels
    ]

    for idx, label_idx in filtered_labels_idx:
        entity = {
            "word": tokenizer.convert_ids_to_tokens(int(model_inputs.input_ids[0][idx])),
            "score": score[0][0][idx][label_idx].item(),
            "entity": id2label[label_idx],
            "index": idx,
        }

        entities += [entity]

    print("Output : ", entities)

    print('Script finished successfully.')

if __name__ == "__main__":
    main()
