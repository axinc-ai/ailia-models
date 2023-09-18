""" from https://github.com/keithito/tacotron """
import utils.g2p.cleaners
from tokenizers import Tokenizer

class PhonemeBpeTokenizer:
  def __init__(self, tokenizer_path = "./utils/g2p/bpe_69.json"):
    self.tokenizer = Tokenizer.from_file(tokenizer_path)

  def tokenize(self, text):
    # 1. convert text to phoneme
    phonemes, langs = _clean_text(text, ['cje_cleaners'])
    # 2. replace blank space " " with "_"
    phonemes = phonemes.replace(" ", "_")
    # 3. tokenize phonemes
    phoneme_tokens = self.tokenizer.encode(phonemes).ids
    assert(len(phoneme_tokens) == len(langs))
    if not len(phoneme_tokens):
      raise ValueError("Empty text is given")
    return phoneme_tokens, langs, phonemes

def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(utils.g2p.cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text, langs = cleaner(text)
  return text, langs
