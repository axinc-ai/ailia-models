import sys
sys.path.append('../onnx_transformers')

from transformers import DistilBertTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_auto import AutoTokenizer

import numpy as np
import time
import sys
import argparse

import ailia

#sys.path.append('../../util')
#from model_utils import check_and_download_models  # noqa: E402

# ======================
# Arguemnt Parser Config
# ======================

DEFAULT_QUESTION = "What is ailia SDK ?"
DEFAULT_CONTEXT = "ailia SDK is a highly performant single inference engine for multiple platforms and hardware"

parser = argparse.ArgumentParser(
    description='bert question answering.'
)

parser.add_argument(
    '--question', '-q', metavar='TEXT',
    default=DEFAULT_QUESTION, 
    help='input question'
)
parser.add_argument(
    '--context', '-c', metavar='TEXT',
    default=DEFAULT_CONTEXT, 
    help='input context'
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

WEIGHT_PATH = "../onnx_transformers/.onnx/deepset/roberta-base-squad2/roberta-base-squad2.onnx"
MODEL_PATH = "../onnx_transformers/.onnx/deepset/roberta-base-squad2/roberta-base-squad2.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_tweets_sentiment/"


# ======================
# Utils
# ======================

# code from https://github.com/patil-suraj/onnx_transformers
from transformers.data import SquadExample, squad_convert_examples_to_features
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy

def create_sample(
    question: Union[str, List[str]], context: Union[str, List[str]]
) -> Union[SquadExample, List[SquadExample]]:
    """
    QuestionAnsweringPipeline leverages the :class:`~transformers.SquadExample` internally.
    This helper method encapsulate all the logic for converting question(s) and context(s) to
    :class:`~transformers.SquadExample`.

    We currently support extractive question answering.

    Arguments:
        question (:obj:`str` or :obj:`List[str]`): The question(s) asked.
        context (:obj:`str` or :obj:`List[str]`): The context(s) in which we will look for the answer.

    Returns:
        One or a list of :class:`~transformers.SquadExample`: The corresponding
        :class:`~transformers.SquadExample` grouping question and context.
    """
    if isinstance(question, list):
        return [SquadExample(None, q, c, None, None, None) for q, c in zip(question, context)]
    else:
        return SquadExample(None, question, context, None, None, None)

def decode(start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int) -> Tuple:
    """
    Take the output of any :obj:`ModelForQuestionAnswering` and will generate probalities for each span to be
    the actual answer.

    In addition, it filters out some unwanted/impossible cases like answer len being greater than
    max_answer_len or answer end position being before the starting position.
    The method supports output the k-best answer through the topk argument.

    Args:
        start (:obj:`np.ndarray`): Individual start probabilities for each token.
        end (:obj:`np.ndarray`): Individual end probabilities for each token.
        topk (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
        max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
    """
    # Ensure we have batch axis
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]

    # Compute the score of each tuple(start, end) to be the real answer
    outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)

    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
    return start, end, candidates[0, start, end]

#@staticmethod
def span_to_answer(tokenizer, text: str, start: int, end: int) -> Dict[str, Union[str, int]]:
    """
    When decoding from token probalities, this method maps token indexes to actual word in
    the initial context.

    Args:
        text (:obj:`str`): The actual context to extract the answer from.
        start (:obj:`int`): The answer starting token index.
        end (:obj:`int`): The answer end token index.

    Returns:
        Dictionary like :obj:`{'answer': str, 'start': int, 'end': int}`
    """
    words = []
    token_idx = char_start_idx = char_end_idx = chars_idx = 0

    for i, word in enumerate(text.split(" ")):
        token = tokenizer.tokenize(word)

        # Append words if they are in the span
        if start <= token_idx <= end:
            if token_idx == start:
                char_start_idx = chars_idx

            if token_idx == end:
                char_end_idx = chars_idx + len(word)

            words += [word]

        # Stop if we went over the end of the answer
        if token_idx > end:
            break

        # Append the subtokenization length to the running index
        token_idx += len(token)
        chars_idx += len(word) + 1

    # Join text with spaces
    return {
        "answer": " ".join(words),
        "start": max(0, char_start_idx),
        "end": min(len(text), char_end_idx),
    }


# ======================
# Main function
# ======================
def main():
    # model files check and download
    #check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # onnx transformers
    from onnx_transformers import pipeline

    inputs = {
      "question": args.question, 
      "context": args.context
    }

    print("Question : ",args.question)
    print("Context : ",args.context)

    # Set defaults values
    handle_impossible_answer = False
    topk = 1
    max_answer_len = 15

    tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')

    # Convert inputs to features
    examples = []

    if True:#for i, item in enumerate(inputs):
        item = inputs
        #print(item)
        if isinstance(item, dict):
            if any(k not in item for k in ["question", "context"]):
                raise KeyError("You need to provide a dictionary with keys {question:..., context:...}")

            example = create_sample(**item)
            examples.append(example)

    features_list = [
        squad_convert_examples_to_features(
            examples=[example],
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            padding_strategy=PaddingStrategy.DO_NOT_PAD.value,
            is_training=False,
            tqdm_enabled=False,
        )
        for example in examples
    ]

    all_answers = []
    for features, example in zip(features_list, examples):
        model_input_names = tokenizer.model_input_names + ["input_ids"]
        fw_args = {k: [feature.__dict__[k] for feature in features] for k in model_input_names}
        fw_args = {k: np.array(v) for (k, v) in fw_args.items()}
        #print("Input",fw_args)
        #print("Shape",fw_args["input_ids"].shape)
        if False:
          net = ailia.Net(MODEL_PATH,WEIGHT_PATH)
          net.set_input_shape(fw_args["input_ids"].shape)
          if args.benchmark:
              print('BENCHMARK mode')
              for i in range(5):
                  start = int(round(time.time() * 1000))
                  outputs = net.predict(fw_args)
                  end = int(round(time.time() * 1000))
                  print("\tailia processing time {} ms".format(end - start))
          else:
            outputs = net.predict(fw_args)
        else:
          from onnxruntime import InferenceSession, SessionOptions, get_all_providers
          from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
          options = SessionOptions()
          options.intra_op_num_threads = 1
          options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
          cpu_model = InferenceSession(WEIGHT_PATH, options, providers=["CPUExecutionProvider"])
          cpu_model.disable_fallback()
          outputs = cpu_model.run(None, fw_args)

        #print("Output",outputs)
        start,end =outputs[0:2]


        min_null_score = 1000000  # large and positive
        answers = []
        for (feature, start_, end_) in zip(features, start, end):
            # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
            undesired_tokens = np.abs(np.array(feature.p_mask) - 1) & feature.attention_mask

            # Generate mask
            undesired_tokens_mask = undesired_tokens == 0.0

            # Make sure non-context indexes in the tensor cannot contribute to the softmax
            start_ = np.where(undesired_tokens_mask, -10000.0, start_)
            end_ = np.where(undesired_tokens_mask, -10000.0, end_)

            # Normalize logits and spans to retrieve the answer
            start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
            end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))

            if handle_impossible_answer:
                min_null_score = min(min_null_score, (start_[0] * end_[0]).item())

            # Mask CLS
            start_[0] = end_[0] = 0.0

            starts, ends, scores = decode(start_, end_, topk, max_answer_len)
            char_to_word = np.array(example.char_to_word_offset)

            # Convert the answer (tokens) back to the original text
            answers += [
                {
                    "score": score.item(),
                    "start": np.where(char_to_word == feature.token_to_orig_map[s])[0][0].item(),
                    "end": np.where(char_to_word == feature.token_to_orig_map[e])[0][-1].item(),
                    "answer": " ".join(
                        example.doc_tokens[feature.token_to_orig_map[s] : feature.token_to_orig_map[e] + 1]
                    ),
                }
                for s, e, score in zip(starts, ends, scores)
            ]

        if handle_impossible_answer:
            answers.append({"score": min_null_score, "start": 0, "end": 0, "answer": ""})

        answers = sorted(answers, key=lambda x: x["score"], reverse=True)[: topk]
        all_answers += answers
    
    print("Answer : ",all_answers)

    print('Script finished successfully.')

if __name__ == "__main__":
    main()
