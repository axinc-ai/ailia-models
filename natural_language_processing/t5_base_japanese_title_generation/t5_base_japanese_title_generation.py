import logging
import re
import sys
import unicodedata

from transformers import T5Tokenizer
import torch
import torch.nn.functional as F
from tqdm import trange
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa

# logger
logger = logging.getLogger(__name__)

"""
params
"""

HUGGING_FACE_MODEL_PATH = "sonoisa/t5-base-japanese-title-generation"
ENCODER_ONNX_PATH = "./t5-base-japanese-title-generation-encoder.onnx"
ENCODER_PROTOTXT_PATH = "./t5-base-japanese-title-generation-encoder.onnx.prototxt"
DECODER_ONNX_PATH = "./t5-base-japanese-title-generation-decoder-with-lm-head.onnx"
DECODER_PROTOTXT_PATH = "./t5-base-japanese-title-generation-decoder-with-lm-head.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/t5_base_japanese_title_generation/"

MAX_SOURCE_LENGTH = 512
DEFAULT_INPUT_PATH = "./input.txt"
DEFAULT_OUTPUT_PATH = "./output.txt"


"""
top k top p filtering algorithm
"""
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Function created by Thomas Wolf of the huggingface team
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

"""
model wrapper
"""
class T5Model(torch.nn.Module):
    """
    This class is based on `GenerativeT5` from `models` in `onnxt5`.
    Modified by Takumi Ibayashi.
    """
    def __init__(self, encoder, decoder_with_lm_head, tokenizer):
        super().__init__()
        self.encoder = encoder
        self.decoder_with_lm_head = decoder_with_lm_head
        self.tokenizer = tokenizer

    def forward(
        self, prompt: str, max_length: int, temperature:float=1.0, repetition_penalty:float=1.0, top_k:int=50, top_p:int=0, max_context_length: int=512, 
    ):
        """
        Generate a text output given a prompt using the model.

        Args:
            prompt (str): The initial text input to the model, which it uses as a 
                starting point to generate the subsequent text.
            max_length (int): The maximum length of the text to be generated.
            temperature (float, optional): This controls the randomness in the model's 
                text generation. A higher temperature value results in more random output. 
                If the temperature is very small, it will approach greedy decoding. 
                Defaults to 1.0.
            repetition_penalty (float, optional): This increases the model's likelihood 
                to generate diverse output by discouraging it from repeating the same 
                token. Defaults to 1.0.
            max_context_length (int, optional): The maximum length of the context to be 
                used in generation. Defaults to 512.
        """
        with torch.no_grad():
            new_tokens = torch.tensor(())
            new_logits = []

            # generate tokens with tokenizer
            token = torch.tensor(self.tokenizer(prompt)['input_ids'])[:max_context_length - 1].unsqueeze(0)

            # encode tokens
            encoder_outputs_prompt = self.encoder.run(None, {"input_ids": token.cpu().numpy()})[0]

            # reset token
            token = torch.zeros((1,1), dtype=torch.long)
            for _ in trange(max_length):
                # decode tokens
                outputs = torch.tensor(
                    self.decoder_with_lm_head.run(
                        None,
                        {"input_ids": token.cpu().numpy(),"encoder_hidden_states": encoder_outputs_prompt},
                    )[0][0]
                )
                next_token_logits = outputs[-1, :] / (temperature if temperature > 0 else 1.0)

                # `1` means end of sentence. (EOS token)
                if int(next_token_logits.argmax()) == 1:
                    break

                new_logits.append(next_token_logits)
                for _ in set(token.view(-1).tolist()):
                    next_token_logits[_] /= repetition_penalty

                # select next token
                if temperature == 0:
                    # greedy sampling: this methods always choose the most probable token.
                    next_token = torch.argmax(next_token_logits).unsqueeze(0)
                else:
                    # Top-k and top-p filtering: this methods enhance text diversity and creativity.
                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                token = torch.cat((token, next_token.unsqueeze(0)), dim=1)
                new_tokens = torch.cat((new_tokens, next_token), 0)
            return self.tokenizer.decode(new_tokens), new_logits

"""
pre process functions
"""
def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

def normalize_text(text: str) -> str:
    assert "\n" not in text and "\r" not in text
    text = text.replace("\t", " ")
    text = text.strip()
    text = normalize_neologd(text)
    text = text.lower()
    return text

def preprocess_body(text: str) -> str:
    return normalize_text(text.replace("\n", " "))

"""
parse args
"""
parser = get_base_parser(
    description="T5 base Japanese title generation",
    default_input=DEFAULT_INPUT_PATH,
    default_save=DEFAULT_OUTPUT_PATH,
    input_ftype="text",
)
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
args = update_parser(parser)

def main(args):
    # download onnx and prototxt
    check_and_download_models(ENCODER_ONNX_PATH, ENCODER_PROTOTXT_PATH, REMOTE_PATH)
    check_and_download_models(DECODER_ONNX_PATH, DECODER_PROTOTXT_PATH, REMOTE_PATH)

    # load model
    tokenizer = T5Tokenizer.from_pretrained(HUGGING_FACE_MODEL_PATH, is_fast=True)
    if args.onnx:
        from onnxruntime import InferenceSession
        encoder_sess = InferenceSession(ENCODER_ONNX_PATH)
        decoder_sess = InferenceSession(DECODER_ONNX_PATH)
        model = T5Model(encoder_sess, decoder_sess, tokenizer)
    else:
        raise Exception("Only onnx runtime mode is supported. Please specify -o option to use onnx runtime.")
        # import ailia
        # import torch
        # encoder_sess = ailia.Net(ENCODER_PROTOTXT_PATH, ENCODER_ONNX_PATH)
        # decoder_sess = ailia.Net(DECODER_PROTOTXT_PATH, DECODER_ONNX_PATH)
        # generated = torch.tensor(tokenizer(prompt)['input_ids'])[:MAX_SOURCE_LENGTH - 1].unsqueeze(0)
        # model = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)

    for input_path in args.input:
        # load input file
        with open(input_path, "r") as fi:
            body = fi.read()

        # pre process
        body_preprocessed = preprocess_body(body)

        # execute prediction
        most_plausible_title, _ = model(body_preprocessed, 21, temperature=1.0, top_k=50, top_p=0.3)
        logger.info("title: %s", most_plausible_title)
        save_path = get_savepath(args.savepath, input_path)
        with open(save_path, "a") as fo:
            fo.write(most_plausible_title)


if __name__ == '__main__':
    main(args)
