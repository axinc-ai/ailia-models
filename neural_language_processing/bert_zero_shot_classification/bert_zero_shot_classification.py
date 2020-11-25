from transformers import AutoTokenizer

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

SENTENCE = "Who are you voting for in 2020?"
CANDIDATE_LABELS = "economics, politics, public health"

parser = argparse.ArgumentParser(
    description='bert zero-shot-classification.'
)

parser.add_argument(
    '--sentence', '-s', metavar='TEXT',
    default=SENTENCE, 
    help='input text'
)
parser.add_argument(
    '--candidate_labels', '-c', metavar='TEXT',
    default=CANDIDATE_LABELS, 
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

WEIGHT_PATH = "roberta-large-mnli.onnx"
MODEL_PATH = "roberta-large-mnli.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_zero_shot_classification/"


# ======================
# Main function
# ======================

def preprocess(sequences, labels, hypothesis_template):
    if len(labels) == 0 or len(sequences) == 0:
        raise ValueError("You must include at least one label and at least one sequence.")
    if hypothesis_template.format(labels[0]) == hypothesis_template:
        raise ValueError(
            (
                'The provided hypothesis_template "{}" was not able to be formatted with the target labels. '
                "Make sure the passed template includes formatting syntax such as {{}} where the label should go."
            ).format(hypothesis_template)
        )

    if isinstance(sequences, str):
        sequences = [sequences]

    sequence_pairs = []
    for sequence in sequences:
        sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])

    return sequence_pairs

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    candidate_labels = CANDIDATE_LABELS.split(", ")

    ailia_model = ailia.Net(MODEL_PATH,WEIGHT_PATH)
    tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')

    hypothesis_template="This example is {}."

    #https://github.com/patil-suraj/onnx_transformers/blob/master/onnx_transformers/pipelines.py
    model_inputs = preprocess(args.sentence,candidate_labels,hypothesis_template)

    #model_inputs = tokenizer.encode_plus(model_inputs, return_tensors="pt")

    model_inputs = tokenizer(
        model_inputs,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
        truncation="only_first",
    )

    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

    print("Sentence : ", args.sentence)
    print("Candidate Labels : ", args.candidate_labels)
    print("Hypothesis Template : ", hypothesis_template)

    print("Inputs : ",inputs_onnx)

    # inference
    if True:
        ailia_model.set_input_blob_shape(inputs_onnx["input_ids"].shape,ailia_model.find_blob_index_by_name("input_ids"))
        ailia_model.set_input_blob_shape(inputs_onnx["attention_mask"].shape,ailia_model.find_blob_index_by_name("attention_mask"))

        if args.benchmark:
            print('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                score = ailia_model.predict(inputs_onnx)
                end = int(round(time.time() * 1000))
                print("\tailia processing time {} ms".format(end - start))
        else:
            score = ailia_model.predict(inputs_onnx)
    else:
        from onnxruntime import InferenceSession, SessionOptions, get_all_providers
        from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_model = InferenceSession(WEIGHT_PATH, options, providers=["CPUExecutionProvider"])
        cpu_model.disable_fallback()
        score = cpu_model.run(None, inputs_onnx)
        print(score)

    score = score[0]
    num_sequences = 1
    reshaped_outputs = score.reshape((num_sequences, len(candidate_labels), -1))

    entail_logits = reshaped_outputs[..., -1]
    score = numpy.exp(entail_logits) / numpy.exp(entail_logits).sum(-1, keepdims=True)

    #score = numpy.exp(score) / numpy.exp(score).sum(-1, keepdims=True)

    label_id=numpy.argmax(numpy.array(score))
    print("Label Id :",label_id)
    print("Label : ",candidate_labels[label_id])
    print("Score : ",score[0][label_id])

    print('Script finished successfully.')

if __name__ == "__main__":
    main()
