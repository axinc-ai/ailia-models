import time
import sys
import os

sys.path.append('../../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

import torch
import json
import numpy as np
import multiprocessing
import random

from pathlib import Path

import transformers

from transformers import AutoModelForCausalLM
from transformers import T5Tokenizer
from transformers import convert_graph_to_onnx

# ======================
# Arguemnt Parser Config
# ======================

DEFAULT_TEXT = '生命、宇宙、そして万物についての究極の疑問の答えは'

parser = get_base_parser('rinna-gpt2 text generation', None, None)
# overwrite
parser.add_argument(
    '--input', '-i', default=DEFAULT_TEXT
)
parser.add_argument(
    '--outlength', '-o', default=50
)
parser.add_argument(
    '-m', '--model_name',
    default='small',
    help='[small, medium]'
)
args = update_parser(parser, check_input_type=False)

def test():
    random.seed(42)

    model_id = "rinna/japanese-gpt2-"+args.model_name
    logger.info(model_id)

    tokenizer = T5Tokenizer.from_pretrained(model_id)
    pt_model = AutoModelForCausalLM.from_pretrained(model_id)

    prompt = args.input
    
    pt_tensor = tokenizer(prompt, return_tensors="pt")["input_ids"]
    output_sequences = pt_model.generate(
        input_ids=pt_tensor,
        max_length=50+pt_tensor.size(1),
        top_p=0.95,
        top_k=50,
        do_sample=True,
        early_stopping=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1
    )

    generated = output_sequences.tolist()[0]
    generated = tokenizer.decode(generated)

    print("pytorch:   ", generated)
   

if __name__ == "__main__":
    test()
