import os
import sys
import time
import numpy as np

import tokenizers
from llama_util import *

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Arguemnt Parser Config
# ======================

DEFAULT_TEXT = 'My name is Clara and I am'


parser = get_base_parser('llama text generation', None, None)
# overwrite

#parser.add_argument('onnxdir', help='llama 7B onnx model directory.')


parser.add_argument(
    '--input', '-i', default=DEFAULT_TEXT
)
parser.add_argument(
    '--model', '-m', type=str,default="llama"
)
parser.add_argument(
    '--temperature',
    default=0.1,
    type=float,
    help=
    'factor to scale up logits, 1.0 means no warp. use `0.1` by default.')
parser.add_argument(
    '--topk',
    default=40,
    type=int,
    help=
    'filter k high score values from logits, None means no filter. 40 by default.'
)
parser.add_argument(
    '--max',
    default=50,
    type=int,
    help=
    'stop condition. default value is 2000, it would stop until len(output_token)==2000.'
)
parser.add_argument(
    '--poolsize',
    default=32,
    type=float,
    help='ailia memory pool size. default value is 32GB')
parser.add_argument('--fp16',
                    action='store_true',
                    help='enable fp16 inference, default True.')
parser.add_argument(
    '--outlength', '-o', default=30
)

parser.add_argument('--length', type=int, default=100, help='max output length.')
parser.add_argument('--n_layer', type=int, default=24, help='layer number, use 24 by default.')
parser.add_argument('--n_embd', type=int, default=1024, help='embedding length, use 1024 by default.')
 
args = parser.parse_args()


args = update_parser(parser, check_input_type=False)

# ======================
# PARAMETERS
# ======================
LLAMA_HEAD_WEIGHT_PATH = "head.onnx"
LLAMA_HEAD_MODEL_PATH = "head.onnx.prototxt"
LLAMA_NORM_WEIGHT_PATH = "norm.onnx"
LLAMA_NORM_MODEL_PATH  = "norm.onnx.prototxt"
LLAMA_EMBED_WEIGHT_PATH = "embed.onnx"
LLAMA_EMBED_MODEL_PATH  = "embed.onnx.prototxt"

LLAMA_DECODER_WEIGHT_PATH = "decoder-merge-"
LLAMA_DECODER_MODEL_PATH  = "decoder-merge-"

RWKV_HEAD_WEIGHT_PATH = "head.onnx"
RWKV_HEAD_MODEL_PATH = "head.onnx.prototxt"
RWKV_EMBED_WEIGHT_PATH = "embed.onnx"
RWKV_EMBED_MODEL_PATH  = "embed.onnx.prototxt"

RWKV_DECODER_WEIGHT_PATH = "mixing_"
RWKV_DECODER_MODEL_PATH  = "mixing_"



REMOTE_PATH = "https://storage.googleapis.com/ailia-models/llama/"

# ======================
# Main function
# ======================

class Llama:
    def __init__(self, onnxdir='models', config: dict = {}):
        if not os.path.exists(onnxdir):
            logger.error('{} not exist'.format(onnxdir))

        assert os.path.isdir(onnxdir)

        self.DECODER_COUNT = 32
        # EOS token
        self.FINISH_TOKEN = 2
        self.tokenizer = Tokenizer(os.path.join(onnxdir, 'tokenizer.model'))

        pool = MemoryPoolSimple(config['poolsize'], args.env_id)
        self.decoder = Decoder(pool, onnxdir, 'decoder-merge-{}.onnx',
                               self.DECODER_COUNT)
        self.config = config

        # cache
        self.pastkeys = [None for i in range(self.DECODER_COUNT)]
        self.pastvalues = [None for i in range(self.DECODER_COUNT)]

        pool.check()

    # Modified transformers.models.llama.modeling_llama._make_causal_mask with np.array
    def _make_causal_mask(self,
                          input_ids_shape,
                          dtype,
                          past_key_values_length: int = 0):
        """    
        Make causal mask used for bi-directional self-attention. 
        Output triangle-matrix if `past_key_values_length`=0
        Padding left if `past_key_values_length`>0
        """
        bsz, tgt_len = input_ids_shape
        mask = np.full((tgt_len, tgt_len), fill_value=np.finfo(dtype).min)

        mask_cond = np.arange(mask.shape[1])
        cond = mask_cond < (mask_cond + 1).reshape(-1, 1)
        mask = np.ma.array(mask, mask=cond, fill_value=0).filled()

        if past_key_values_length > 0:
            mask = np.concatenate([
                np.zeros((tgt_len, past_key_values_length), dtype=dtype), mask
            ],
                                  axis=1)

        return mask.reshape(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    # Modified transformers.models.llama.modeling_llama._expand_mask with np.array
    def _expand_mask(self, mask, dtype, tgt_len=None):
        """  
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.  
        """
        bsz, src_len = mask.shape
        if tgt_len is None:
            tgt_len = src_len
        # expand [bsz,38] to [bsz,1,1,38]
        expanded_mask = np.expand_dims(mask, axis=1)
        expanded_mask = np.expand_dims(mask, axis=1)
        expanded_mask = np.broadcast_to(expanded_mask,
                                        (bsz, 1, tgt_len, src_len))
        inverted_mask = 1.0 - expanded_mask

        cond = inverted_mask > 0
        return np.ma.array(inverted_mask,
                           mask=cond,
                           fill_value=np.finfo(dtype).min).filled()

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask,
                                                   inputs_embeds.dtype,
                                                   tgt_len=input_shape[-1])
            combined_attention_mask = (expanded_attn_mask
                                       if combined_attention_mask is None else
                                       expanded_attn_mask +
                                       combined_attention_mask)

        return combined_attention_mask

    def convert_to_fp16(self, inputs):
        outputs = dict()
        for k, v in inputs.items():
            if v.dtype == np.float32:
                outputs[k] = v.astype(np.float16)
            else:
                outputs[k] = v
        return outputs

    def decode(self, token: np.array):
        # embed space
        hidden = self.decoder.embed(token)
        assert hidden.shape[-1] == 4096

        if self.pastkeys[0] is None:
            pastlen = 0
        else:
            pastlen = self.pastkeys[0].shape[-2]
        seqlen = hidden.shape[1]

        position_ids = np.arange(seqlen, dtype=np.int64).reshape((1, seqlen))
        position_ids[0][0] = pastlen

        attention_mask = np.ones((1, seqlen + pastlen), dtype=np.float32)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (1, seqlen), hidden, pastlen)

        for idx in range(self.DECODER_COUNT):
            past_key = self.pastkeys[idx]
            past_value = self.pastvalues[idx]

            if past_key is None:
                zero_tensor = np.zeros((1, 32, 0, 128), dtype=np.float32)
                inputs = {
                    'hidden_in': hidden,
                    'attn_mask': attention_mask,
                    'position_ids': position_ids,
                    'past_key_in': zero_tensor,
                    'past_value_in': zero_tensor
                }
            else:
                inputs = {
                    'hidden_in': hidden,
                    'attn_mask': attention_mask,
                    'position_ids': position_ids,
                    'past_key_in': past_key,
                    'past_value_in': past_value
                }

            if self.config['fp16']:
                inputs = self.convert_to_fp16(inputs)
            outputs = self.decoder.decode(inputs, idx)

            hidden = outputs[0]  # [[[ 0.0221,  0.0120,  0.0007,  ..., -0.0614, -0.0625,  0.0494]]]
            self.pastkeys[idx] = outputs[1]
            self.pastvalues[idx] = outputs[2]

        hidden = self.decoder.norm_head(hidden)
        return hidden

    def apply_warp(self, tensor: np.array):
        tensor = warp_temperature(tensor, self.config['temperature'])
        tensor = warp_topk(tensor, self.config['topk'])
        return tensor

    def sample(self, prompt: str = 'bonjour'):
        PROMPT_DICT = {
            "prompt_no_input":
            ("Below is an instruction that describes a task. "
             "Write a response that appropriately completes the request.\n\n"
             "### Instruction:\n{instruction}\n\n### Response:"),
        }
        PROMPT = PROMPT_DICT['prompt_no_input']

        prompt = prompt.strip()
        format_prompt = PROMPT.format_map({'instruction': prompt})

        # no EOS
        input_ids = self.tokenizer.encode(format_prompt, True, False)
        input_ids = np.array(input_ids, dtype=np.int64).reshape(
            (1, len(input_ids)))

        # decoder backbone loop
        next_token = input_ids
        pre = 0
        
        while True:
            # decoder backbone
            logits = self.decode(next_token)

            # split tail
            next_token_scores = logits[:, -1, :]

            # wrap logits for better token
            next_token_scores = self.apply_warp(next_token_scores)

            probs = npsoftmax(next_token_scores.astype(np.float64), axis=1)

            # Caution:
            # *** ValueError: sum(pvals[:-1].astype(np.float64)) > 1.0. The pvals array is cast to 64-bit floating point prior to checking the sum. Precision changes when casting may cause problems even if the sum of the original pvals is valid.
            next_token = npmultinominal2D(probs).astype(input_ids.dtype)

            input_ids = np.concatenate(
                [input_ids, next_token.reshape((1, 1))], axis=1)


            decoded = self.tokenizer.decode(input_ids[0].tolist())
            out = str(decoded.split('Response:')[1])

            # stream print
            now = len(out)
            if now - 1 > pre:
                #print(out[pre: now-1], end="", flush=True)
                pre = now - 1

            if input_ids.shape[-1] >= self.config['max'] or next_token[
                    0, 0] == self.FINISH_TOKEN:
                break

        # decode
        decoded = self.tokenizer.decode(input_ids[0].tolist())
        out = str(decoded.split('Response:')[1]).replace("###","")
        return out

class RWKV_RNN():

    def __init__(self, onnxdir: str, n_layer=24):
        self.embed = OrtWrapper('embed_rwkv.onnx', env_id = args.env_id)
        self.head = OrtWrapper('head_rwkv.onnx', env_id = args.env_id)
        self.backbone = []
        for i in range(n_layer):
            self.backbone.append(OrtWrapper(os.path.join(onnxdir, 'mixing_{}_rwkv.onnx'.format(i)), env_id = args.env_id))

    def forward(self, token, state):
        token = np.full((1), token, dtype=np.int32)
        x = self.embed.forward({'token': token})[0]

        for i, node in enumerate(self.backbone):
            state_in = state[5 * i:5 * i + 5]
            out = node.forward({'input': x.astype(np.float16), 'state_in': state_in})
            x = out[0]
            state[5 * i:5 * i + 5] = out[1]

        return self.head.forward({'x': x.astype(np.float16)})[0], state


def llama_main():
    logger.info("This model requires multiple input shape, so running on CPU")
    logger.info("Input : "+args.input)
    llama = Llama(onnxdir="./",
                  config={
                      'temperature': args.temperature,
                      'topk': args.topk,
                      'max': args.max,
                      'poolsize': args.poolsize,
                      'fp16': args.fp16
                  })
    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            output = llama.sample(args.input)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        output = llama.sample(args.input)

    logger.info("output : "+output)
    logger.info('Script finished successfully.')

def rwkv_main():
    logger.info("This model requires multiple input shape, so running on CPU")
    logger.info("Input : "+args.input)

    tokenizer = tokenizers.Tokenizer.from_file("20B_tokenizer.json")

    #context = "\nIn a shocking finding, "


    def compute(context):
        model = RWKV_RNN("./", n_layer=args.n_layer)
        init_state = np.zeros((args.n_layer * 5, args.n_embd), dtype=np.float16)
         
        #print('\nPreprocessing context. {}'.format(context))
        for token in tokenizer.encode(context).ids:
            init_out, init_state = model.forward(token, init_state)
            #print('.', end="", flush=True)

        all_tokens = []
        out_last = 0
        out, state = init_out, init_state
        output = ""
        for i in range(args.length):
            token = sample_logits(out.astype(np.float32))
            all_tokens += [token]
            tmp = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp:  # only print when we have a valid utf-8 string
                #print(tmp, end="", flush=True)
                out_last = i + 1
            output += tmp
            out, state = model.forward(token, state)
        return output
    

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            output = compute(args.input)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        output = compute(args.input)

    logger.info("output : "+output)
    logger.info('Script finished successfully.')


if __name__ == "__main__":
    # model files check and download
    check_and_download_models(LLAMA_HEAD_WEIGHT_PATH, LLAMA_HEAD_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(LLAMA_NORM_WEIGHT_PATH, LLAMA_NORM_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(LLAMA_EMBED_WEIGHT_PATH,LLAMA_EMBED_MODEL_PATH, REMOTE_PATH)

    args.env_id = -1
    logger.warning("This model requires too large memory. So we force use cpu.")

    if args.model == "llama":
        fp16 = ""
        if args.fp16:
            fp16 = "-fp16"
        for i in range(32):
            WEIGHT_PATH = LLAMA_DECODER_WEIGHT_PATH + str(i) + fp16 +".onnx"
            MODEL_PATH  = LLAMA_DECODER_WEIGHT_PATH + str(i) + fp16 +".onnx.prototxt"
            check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
        llama_main()
    else:
        check_and_download_models(RWKV_HEAD_WEIGHT_PATH, RWKV_HEAD_MODEL_PATH, REMOTE_PATH)
        check_and_download_models(RWKV_EMBED_WEIGHT_PATH,RWKV_EMBED_MODEL_PATH, REMOTE_PATH)
        for i in range(23):
            WEIGHT_PATH = RWKV_DECODER_WEIGHT_PATH + str(i) +"_rwkv.onnx"
            MODEL_PATH  = RWKV_DECODER_WEIGHT_PATH + str(i) +"_rwkv.onnx.prototxt"
            check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
        rwkv_main()

