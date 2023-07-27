import sys
import time
from typing import Union, Tuple
from logging import getLogger

import numpy as np
from transformers import MarianTokenizer
import onnxruntime
from optimum.onnxruntime import ORTModelForCausalLM
import ailia


ORTModelForCausalLM.from_pretrained

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
HUGGING_FACE_PATH = 'staka/fugumt-ja-en'
ENCODER_ONNX_PATH  = 'encoder_model.onnx'
ENCODER_PROTOTXT_PATH = 'encoder_model.onnx.prototxt'
DECODER_ONNX_PATH  = 'decoder_model_merged.onnx'
DECODER_PROTOTXT_PATH = 'decoder_model_merged.onnx.prototxt'
# REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/fugumt/'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'FuguMT', None, None
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default="これは猫です",
    help="Input text."
)
parser.add_argument(
    '--onnx',
    '-o',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser, check_input_type=False)
args.onnx = True

# ======================
# Model wrapper
# ======================
class MarianMT:
    def __init__(
        self,
        tokenizer: MarianTokenizer,
        encoder: Union[ailia.Net, onnxruntime.InferenceSession],
        decoder: Union[ailia.Net, onnxruntime.InferenceSession],
        args,
    ) -> None:
        if type(encoder) not in [ailia.Net, onnxruntime.InferenceSession]:
            raise Exception(f"encoder type {type(encoder)} is not supported")
        if type(decoder) not in [ailia.Net, onnxruntime.InferenceSession]:
            raise Exception(f"decoder type {type(decoder)} is not supported")
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        decoder_input_ids: np.ndarray,
        past_key_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """execute encoder and decoder once and return response of decoder

        Args:
            input_ids (np.ndarray):
            attention_mask (np.ndarray):
            decoder_input_ids (np.ndarray):
            past_key_values (np.ndarray):

        Returns:
            Tuppe[np.ndarray, np.ndarray]: logits and new past_key_values
        """
        def _decode_with_cache(
            self,
            attention_mask: np.ndarray,
            decoder_input_ids: np.ndarray,
            past_key_values: np.ndarray,
        ):
            return self.decoder.run(
                None,
                {
                    'input_ids': decoder_input_ids[:, -1:], # only input last token
                    'encoder_attention_mask': attention_mask,
                    'encoder_hidden_states': encoder_output[0],
                    'use_cache_branch': np.array([True]),
                    'past_key_values.0.decoder.key': past_key_values[0],
                    'past_key_values.0.decoder.value': past_key_values[1],
                    'past_key_values.0.encoder.key': past_key_values[2],
                    'past_key_values.0.encoder.value': past_key_values[3],
                    'past_key_values.1.decoder.key': past_key_values[4],
                    'past_key_values.1.decoder.value': past_key_values[5],
                    'past_key_values.1.encoder.key': past_key_values[6],
                    'past_key_values.1.encoder.value': past_key_values[7],
                    'past_key_values.2.decoder.key': past_key_values[8],
                    'past_key_values.2.decoder.value': past_key_values[9],
                    'past_key_values.2.encoder.key': past_key_values[10],
                    'past_key_values.2.encoder.value': past_key_values[11],
                    'past_key_values.3.decoder.key': past_key_values[12],
                    'past_key_values.3.decoder.value': past_key_values[13],
                    'past_key_values.3.encoder.key': past_key_values[14],
                    'past_key_values.3.encoder.value': past_key_values[15],
                    'past_key_values.4.decoder.key': past_key_values[16],
                    'past_key_values.4.decoder.value': past_key_values[17],
                    'past_key_values.4.encoder.key': past_key_values[18],
                    'past_key_values.4.encoder.value': past_key_values[19],
                    'past_key_values.5.decoder.key': past_key_values[20],
                    'past_key_values.5.decoder.value': past_key_values[21],
                    'past_key_values.5.encoder.key': past_key_values[22],
                    'past_key_values.5.encoder.value': past_key_values[23],
                }
            )
        def _decode_without_cache(
            self,
            attention_mask: np.ndarray,
            decoder_input_ids: np.ndarray,
            past_key_values: np.ndarray,
        ):
            return self.decoder.run(
                None,
                {
                    'input_ids': decoder_input_ids, # only input last token
                    'encoder_attention_mask': attention_mask,
                    'encoder_hidden_states': encoder_output[0],
                    'use_cache_branch': np.array([False]),
                    'past_key_values.0.decoder.key': past_key_values[0],
                    'past_key_values.0.decoder.value': past_key_values[1],
                    'past_key_values.0.encoder.key': past_key_values[2],
                    'past_key_values.0.encoder.value': past_key_values[3],
                    'past_key_values.1.decoder.key': past_key_values[4],
                    'past_key_values.1.decoder.value': past_key_values[5],
                    'past_key_values.1.encoder.key': past_key_values[6],
                    'past_key_values.1.encoder.value': past_key_values[7],
                    'past_key_values.2.decoder.key': past_key_values[8],
                    'past_key_values.2.decoder.value': past_key_values[9],
                    'past_key_values.2.encoder.key': past_key_values[10],
                    'past_key_values.2.encoder.value': past_key_values[11],
                    'past_key_values.3.decoder.key': past_key_values[12],
                    'past_key_values.3.decoder.value': past_key_values[13],
                    'past_key_values.3.encoder.key': past_key_values[14],
                    'past_key_values.3.encoder.value': past_key_values[15],
                    'past_key_values.4.decoder.key': past_key_values[16],
                    'past_key_values.4.decoder.value': past_key_values[17],
                    'past_key_values.4.encoder.key': past_key_values[18],
                    'past_key_values.4.encoder.value': past_key_values[19],
                    'past_key_values.5.decoder.key': past_key_values[20],
                    'past_key_values.5.decoder.value': past_key_values[21],
                    'past_key_values.5.encoder.key': past_key_values[22],
                    'past_key_values.5.encoder.value': past_key_values[23],
                }
            )
        # execute encoder
        encoder_output = self.encoder.run(
            None,
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            },
        )

        # execute decoder
        try:
            decoder_output = _decode_with_cache(
                self,
                attention_mask,
                decoder_input_ids,
                past_key_values,
            )
        except Exception:
            # HACK: If all past_key_values is not returned, there will be an exception, so retry without using the cache.
            decoder_output = _decode_without_cache(
                self,
                attention_mask,
                decoder_input_ids,
                past_key_values,
            )
        logits, new_past_key_values = decoder_output[0], decoder_output[1:]
        return logits, new_past_key_values

    def recognize_from_text(self, input_text: str):
        ## greedy search
        model_inputs = self.tokenizer(input_text)
        output_ids = self._greedy_search(model_inputs)
        translation_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return translation_text

    def _greedy_search(self, model_inputs: dict):
        """
        reference:
        https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L2289
        """
        decoder_start_token_id = pad_token_id = 32000
        eos_token_id = 0

        input_ids = np.array([model_inputs["input_ids"]] * 1, dtype=int)
        attention_mask = np.array([model_inputs["attention_mask"]] * 1, dtype=int)
        decoder_input_ids = np.ones((1, 1), dtype=int) * decoder_start_token_id
        past_key_values = [np.zeros((1, 8, 0, 64), dtype=np.float32)] * 24

        # keep track of which sequences are already finished
        unfinished_sequences = np.full(decoder_input_ids.shape[0], 1, dtype=input_ids.dtype)
        while True:
            logits, past_key_values = self.forward(input_ids, attention_mask, decoder_input_ids, past_key_values)
            next_token_logits = logits[:, -1, :]
            next_token_logits[:, pad_token_id] = float("-inf") # fix: pad_token_id must not be selected as next token
            next_token_scores_processed = self._logits_processor(decoder_input_ids, next_token_logits)

            # select token with has maximum score as next token (greedy selection)
            next_tokens = np.argmax(next_token_scores_processed, axis=-1)

            # If unfinished_sequences is 0, replace it with pad_token_id.
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # append next token to decoder_input_ids
            decoder_input_ids = np.concatenate([decoder_input_ids, np.expand_dims(next_tokens, axis=-1)], axis=-1)

            # if eos_token was found in one sentence, set sentence to finished
            num_not_eos = sum(next_tokens != eos_token_id)
            unfinished_sequences = unfinished_sequences * num_not_eos

            # stop when there is a </s> in each estimated sentence
            if unfinished_sequences.max() == 0:
                break
        return decoder_input_ids
   
    def _logits_processor(self, input_ids, scores):
        max_length = 512
        eos_token_id = [0]

        cur_len = input_ids.shape[-1]
        if cur_len == max_length - 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i not in eos_token_id]] = -float("inf")
            for i in eos_token_id:
                scores[:, i] = 0
        
        return scores

def main():
    # model files check and download
    # check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    tokenizer = MarianTokenizer.from_pretrained(HUGGING_FACE_PATH)
    env_id = args.env_id

    if not args.onnx:
        env_id = args.env_id
        encoder = ailia.Net(ENCODER_ONNX_PATH, ENCODER_PROTOTXT_PATH, env_id=env_id)
        decoder = ailia.Net(DECODER_ONNX_PATH, DECODER_PROTOTXT_PATH, env_id=env_id)
    else:
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        encoder = onnxruntime.InferenceSession(ENCODER_ONNX_PATH, providers=providers)
        decoder = onnxruntime.InferenceSession(DECODER_ONNX_PATH, providers=providers)

    input_text = args.input
    logger.info("input_text: %s" % input_text)

    # inference
    logger.info('Start inference...')
    model = MarianMT(tokenizer, encoder, decoder, args)

    if args.benchmark:
        raise Exception("not supported yet")
    else:
        output = model.recognize_from_text(input_text)

    logger.info(f"translation_text: {output}")
    logger.info('Script finished successfully.')

if __name__ == '__main__':
    main()





