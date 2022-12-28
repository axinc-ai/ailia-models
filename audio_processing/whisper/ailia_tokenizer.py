import os

import numpy as np
from typing import List, Optional, Tuple, Union

import json

from languages import LANGUAGES, TO_LANGUAGE_CODE

def get_tokenizer(multilingual: bool,
        task: Optional[str] = None,  # Literal["transcribe", "translate", None]
        language: Optional[str] = None):
    if multilingual:
        tokenizer_name = "multilingual"
        task = task or "transcribe"
        language = language or "en"
    else:
        tokenizer_name = "gpt2"
        task = None
        language = None

    tokenizer = AiliaTokenizer()
    tokenizer.build_tokenizer('assets/multilingual/vocab.json', tokenizer_name, task, language)
    return tokenizer

class AiliaTokenizer:
    # dictionary
    vocab = {}
    byte_encoder = None
    byte_decode = None

    # state
    language = None

    # sequence
    sot_sequence = None
    all_language_tokens_list = None

    # special tokens
    sot = None
    sot_prev = None
    sot_lm = None
    no_speech = None
    eot = None
    no_timestamps = None
    translate = None
    transcribe = None
    timestamp_begin = None

    def build_tokenizer(self, vocab_path, tokenizer_name, task, language):
        self.language = language

        # load vocab
        json_open = open(vocab_path, 'r', encoding='utf-8')
        json_load = json.load(json_open)
        for key in json_load.keys():
            self.vocab[json_load[key]] = key
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        #specials = [
        #"<|startoftranscript|>",
        #*[f"<|{lang}|>" for lang in LANGUAGES.keys()],
        #"<|translate|>",
        #"<|transcribe|>",
        #"<|startoflm|>",
        #"<|startofprev|>",
        #"<|nospeech|>",
        #"<|notimestamps|>",
        #]

        # load special token ids
        if tokenizer_name == "multilingual":
            multilingal = 1
        else:
            multilingal = 0
        self.eot  = 50256 + multilingal
        self.sot  = 50257 + multilingal
        language_tokens = []
        i = self.sot + 1
        for lang in LANGUAGES.keys():
            language_tokens.append(i)
            i = i + 1
        self.translate = 50357 + multilingal
        self.transcribe = 50358 + multilingal
        self.sot_lm = 50359 + multilingal
        self.sot_prev = 50360 + multilingal
        self.no_speech = 50361 + multilingal
        self.no_timestamps = 50362 + multilingal
        self.timestamp_begin = 50363 + multilingal

        self.all_language_tokens_list = language_tokens

        langs = tuple(LANGUAGES.keys())
        sot_sequence = [self.sot]
        if language is not None:
            sot_sequence.append(self.sot + 1 + langs.index(language))
        if task is not None:
            sot_sequence.append(self.transcribe if task == "transcribe" else self.translate)
        self.sot_sequence = sot_sequence
        
    def bytes_to_unicode(self):
        """
        Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
        characters the bpe code barfs on.
        The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
        if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
        decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
        tables between utf-8 bytes and unicode strings.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="error")
        return text

    def decode(self, tokens):
        tokens = [token for token in tokens if token < self.eot]
        vocab_tokens = []
        for token in tokens:
            vocab_tokens.append(self.vocab[token])
        return self.convert_tokens_to_string(vocab_tokens)

    @property
    def language_token(self) -> int:
        i = 0
        for lang in LANGUAGES.keys():
            if lang == self.language:
                return self.all_language_tokens_list[i]
            i = i + 1
        raise KeyError(f"Language {self.language} not found in tokenizer.")

    @property
    def all_language_tokens(self) -> Tuple[int]:
        return self.all_language_tokens_list

    @property
    def all_language_codes(self) -> Tuple[str]:
        return tuple(l for l in LANGUAGES.keys())

    @property
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        return (1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254)
