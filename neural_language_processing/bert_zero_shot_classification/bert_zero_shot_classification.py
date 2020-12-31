import time
import sys

from transformers import AutoTokenizer
import numpy

import ailia

sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402


# ======================
# Arguemnt Parser Config
# ======================

SENTENCE = "Who are you voting for in 2020?"
CANDIDATE_LABELS = "economics, politics, public health"
HYPOTHESIS_TEMPLATE = "This example is {}."


parser = get_base_parser('bert zero-shot-classification.', None, None)
parser.add_argument(
    '--sentence', '-s', metavar='TEXT', default=SENTENCE,
    help='input sentence'
)
parser.add_argument(
    '--candidate_labels', '-c', metavar='TEXT', default=CANDIDATE_LABELS,
    help='input labels separated by , '
)
parser.add_argument(
    '--hypothesis_template', '-t', metavar='TEXT', default=HYPOTHESIS_TEMPLATE,
    help='input hypothesis template'
)
args = update_parser(parser)


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = "roberta-large-mnli.onnx"
MODEL_PATH = "roberta-large-mnli.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_zero_shot_classification/"


# ======================
# Main function
# ======================

def preprocess(sequences, labels, hypothesis_template):
    sequences = [sequences]
    sequence_pairs = []
    for sequence in sequences:
        sequence_pairs.extend(
            [[sequence, hypothesis_template.format(label)] for label in labels]
        )

    return sequence_pairs


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    candidate_labels = CANDIDATE_LABELS.split(", ")

    ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')

    model_inputs = preprocess(
        args.sentence, candidate_labels, args.hypothesis_template
    )

    model_inputs = tokenizer(
        model_inputs,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
        truncation="only_first",
    )

    inputs_onnx = {
        k: v.cpu().detach().numpy() for k, v in model_inputs.items()
    }

    print("Sentence : ", args.sentence)
    print("Candidate Labels : ", args.candidate_labels)
    print("Hypothesis Template : ", args.hypothesis_template)

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

    score = score[0]
    num_sequences = 1
    reshaped_outputs = score.reshape(
        (num_sequences, len(candidate_labels), -1)
    )

    entail_logits = reshaped_outputs[..., -1]
    score = numpy.exp(entail_logits) / \
        numpy.exp(entail_logits).sum(-1, keepdims=True)

    label_id = numpy.argmax(numpy.array(score))
    print("Label Id :", label_id)
    print("Label : ", candidate_labels[label_id])
    print("Score : ", score[0][label_id])

    print('Script finished successfully.')


if __name__ == "__main__":
    main()
