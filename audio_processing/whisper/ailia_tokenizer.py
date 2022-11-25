import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np

import json

class AiliaTokenizer:
    vocab = {}
    byte_encoder = None
    byte_decode = None

    def open(self, vocab_path):
        json_open = open(vocab_path, 'r')
        json_load = json.load(json_open)
        #print(json_load)
        for key in json_load.keys():
            #print(json_load[key], key)
            self.vocab[json_load[key]] = key
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

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
        vocab_tokens = []
        for token in tokens:
            vocab_tokens.append(self.vocab[token])
        return self.convert_tokens_to_string(vocab_tokens)
