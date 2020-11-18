import sys
import torch
import numpy
import argparse

from transformers import BertTokenizer, BertJapaneseTokenizer, BertForMaskedLM
from onnxruntime import InferenceSession, SessionOptions, get_all_providers

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

parser = argparse.ArgumentParser(
    description='masklm proofreading sample'
)
parser.add_argument(
    '-i', '--input', metavar='VIDEO',
    default="test_text_jp.txt",
    help='The input video path.'
)
parser.add_argument(
    '-s', '--sudjest',
    action='store_true',
    help='Show sudjestion'
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

    is_english = args.arch=='bert-base-cased' or args.arch=='bert-base-uncased'
    show_sudject = False

    if is_english:
        tokenizer = BertTokenizer.from_pretrained(args.arch)
    else:
        tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/"+args.arch)

    ailia_mode = False
    if ailia_mode:
        import ailia
        cpu_model = ailia.Net(MODEL_PATH,WEIGHT_PATH)
    else:
        cpu_model = InferenceSession(WEIGHT_PATH)

    import codecs

    with codecs.open(args.input, 'r', 'utf-8', 'ignore') as f:
        s = f.readlines()

    for text in s:
        tokenized_text = tokenizer.tokenize(text)
        score = numpy.zeros((len(tokenized_text)))
        sujest = {}

        #print("Tokenized text : ",tokenized_text)

        result_text = tokenizer.tokenize(text)

        for i in range(0,len(tokenized_text)):
            masked_index = i
            tokenized_text_saved = tokenized_text[masked_index] 

            tokenized_text[masked_index] = '[MASK]'
            #print("Masked text : ",tokenized_text)

            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            #print("Indexed tokens : ",indexed_tokens)

            if ailia_mode:
                indexed_tokens = numpy.expand_dims(numpy.array(indexed_tokens), axis=0)
                token_type_ids = numpy.zeros((1,len(tokenized_text)))
                attention_mask = numpy.zeros((1,len(tokenized_text)))
            else:
                indexed_tokens = [numpy.array(indexed_tokens)]
                token_type_ids = [numpy.zeros((len(tokenized_text)))]
                attention_mask = [numpy.zeros((len(tokenized_text)))]

            inputs_onnx = {"token_type_ids":token_type_ids,"input_ids":indexed_tokens,"attention_mask":attention_mask}

            #print("Input : ",inputs_onnx)

            #print("Predicting...")
            if ailia_mode:
                cpu_model.set_input_blob_shape(indexed_tokens.shape,cpu_model.find_blob_index_by_name("token_type_ids"))
                cpu_model.set_input_blob_shape(indexed_tokens.shape,cpu_model.find_blob_index_by_name("input_ids"))
                cpu_model.set_input_blob_shape(attention_mask.shape,cpu_model.find_blob_index_by_name("attention_mask"))
                outputs = cpu_model.predict(inputs_onnx)
            else:
                outputs = cpu_model.run(None, inputs_onnx)

            #print("Output : ",outputs)
            def softmax(x):
                u = numpy.sum(numpy.exp(x))
                return numpy.exp(x)/u

            outputs[0][0, masked_index] = softmax(outputs[0][0, masked_index])

            target_ids = tokenizer.convert_tokens_to_ids([tokenized_text_saved])
            index = target_ids[0]
            score[masked_index] = outputs[0][0, masked_index][index]

            predictions = torch.from_numpy(outputs[0][0, masked_index]).topk(1)
            index = predictions.indices[0]
            top_token = tokenizer.convert_ids_to_tokens([index])[0]
            sujest[masked_index] = top_token

            tokenized_text[masked_index] = tokenized_text_saved

        if is_english:
            space=" "
        else:
            space=""

        fine_text = ""
        for i in range(0,len(tokenized_text)):
            prob_yellow = 0.0001
            prob_red = 0.00001
            if score[i]<prob_red:
                fine_text=fine_text+'\033[31m'+space+tokenized_text[i]+'\033[0m'
                if args.sudjest:
                    fine_text=fine_text+' ->\033[34m'+space+sujest[i]+'\033[0m'
            elif score[i]<prob_yellow:
                fine_text=fine_text+'\033[33m'+space+tokenized_text[i]+'\033[0m'
                if args.sudjest:
                    fine_text=fine_text+' ->\033[34m'+space+sujest[i]+'\033[0m'
            else:
                fine_text=fine_text+space+tokenized_text[i]

        if is_english:
            fine_text = fine_text.replace(' ##', '')
        else:
            fine_text = fine_text.replace('##', '')

        print(fine_text)

if __name__ == "__main__":
    main()
