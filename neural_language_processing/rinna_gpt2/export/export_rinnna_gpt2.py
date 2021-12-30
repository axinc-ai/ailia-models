import time
import sys
import os
from transformers import T5Tokenizer
import numpy

sys.path.append('../../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

import torch
import json
import numpy as np
import onnx
import multiprocessing
import transformers

from transformers import convert_graph_to_onnx
from pathlib import Path

# ======================
# Arguemnt Parser Config
# ======================

DEFAULT_TEXT = '生命、宇宙、そして万物についての究極の疑問の答えは'

parser = get_base_parser('rinna-gpt2 text generation', None, None)
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

def generate_text(tokenizer, ailia_model, span, outputlength):
    model_input = tokenizer.encode_plus(span)
    model_input = {name : np.atleast_2d(value) for name, value in model_input.items()}

    print(model_input)

    model_input['input_ids'] = np.array(model_input['input_ids'], dtype='int64')
    model_input['attention_mask'] = np.array(model_input['attention_mask'], dtype='int64')

    onnx_result = [ailia_model(torch.from_numpy(model_input['input_ids']),torch.from_numpy(model_input['attention_mask'])).detach().numpy()]

    K=outputlength
    predictions = np.argpartition(-onnx_result[0][0, -1], K)[:K]

    out_str = span
    for i in range(outputlength):
      index = predictions[0]

      #next_token_logits = onnx_result[0][:, -1, :]
      #next_tokens = np.argmax(next_token_logits, axis=-1)
      #index = next_tokens[0]

      token = tokenizer.convert_ids_to_tokens([index])[0]
      out_str += token

      #trim = 1
      trim = 0
      
      input = np.append(model_input['input_ids'][:,trim:], index)    
      model_input['input_ids'] = np.expand_dims(input, 0)

      attention_mask = np.append(model_input['attention_mask'][:,trim:], 1)    
      model_input['attention_mask'] = np.expand_dims(attention_mask, 0)

      onnx_result = [ailia_model(torch.from_numpy(model_input['input_ids']),torch.from_numpy(model_input['attention_mask'])).detach().numpy()]

      predictions = np.argpartition(-onnx_result[0][0, -1], K)[:K]

      if token == "<unk>":
        break

    return out_str

def export():
    span = args.input

    model_name = "rinna/japanese-gpt2-"+args.model_name
    model_pth = Path(f"../japanese-gpt2-"+args.model_name+".onnx")

    pipeline_name = "text-generation"

    model_pth.parent.mkdir(exist_ok=True, parents=True)

    nlp = transformers.pipeline(pipeline_name, model=model_name, tokenizer=model_name)
    tokenizer = nlp.tokenizer
    model = nlp.model

    with torch.no_grad():
        (
            input_names,
            output_names,
            dynamic_axes,
            tokens,
        ) = convert_graph_to_onnx.infer_shapes(nlp, "pt")
        ordered_input_names, model_args = convert_graph_to_onnx.ensure_valid_input(
            nlp.model, tokens, input_names
        )

    class GPT2Sent(transformers.GPT2LMHeadModel):
        def __init__(self, config):
            super().__init__(config)
            self.sentence_embedding = torch.nn.Identity()

        def forward(self, input_ids, attention_mask):
            return self.sentence_embedding(
                super().forward(input_ids, attention_mask=attention_mask).logits
            )

    # Create the new model based on the config of the original pipeline
    model = GPT2Sent(config=nlp.model.config).from_pretrained(model_name)

    encoding = nlp.tokenizer([span], return_tensors="pt")
    print(encoding)

    if not model_pth.exists():
        inputs = ['input_ids','attention_mask']
        outputs = ['3062']
        dynamic_axes = {'input_ids': {1: 'len'}, 'attention_mask': {1: 'len'}, '3062': {1: 'len'}}

        torch.onnx.export(
            model,
            (encoding["input_ids"], encoding["attention_mask"]),
            f=model_pth.as_posix(),
            input_names=input_names,
            do_constant_folding=True,
            use_external_data_format=False,
            enable_onnx_checker=True,
            opset_version=11,
            dynamic_axes=dynamic_axes
        )
    
    output = generate_text(nlp.tokenizer, model, args.input, int(args.outlength))
    print(output)

if __name__ == "__main__":
    export()
