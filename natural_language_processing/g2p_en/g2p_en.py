#from nltk import pos_tag
#from nltk.corpus import cmudict
#import nltk
#from nltk.tokenize import TweetTokenizer
#word_tokenize = TweetTokenizer().tokenize
import codecs
import re
import os
import unicodedata
from builtins import str as unicode
from expand import normalize_numbers
import numpy as np
import ailia
from averaged_perceptron import tag

#try:
#    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
#except LookupError:
#    nltk.download('averaged_perceptron_tagger')
#try:
#    nltk.data.find('corpora/cmudict.zip')
#except LookupError:
#    nltk.download('cmudict')

dirname = os.path.dirname(__file__)

def construct_homograph_dictionary():
    f = os.path.join(dirname,'homographs.en')
    homograph2features = dict()
    for line in codecs.open(f, 'r', 'utf8').read().splitlines():
        if line.startswith("#"): continue # comment
        headword, pron1, pron2, pos1 = line.strip().split("|")
        homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)
    return homograph2features

def construct_cmu_dictionary():
    f = os.path.join(dirname,'cmudict')
    cmudict = dict()
    for line in codecs.open(f, 'r', 'utf8').read().splitlines():
        if line.startswith("#"): continue # comment
        lists = line.strip().split(" ")
        headword = lists[0]
        pron = lists[2:]
        key = headword.lower()
        if not (key in cmudict):
            cmudict[key] = pron
    return cmudict

class G2p(object):
    def __init__(self):
        super().__init__()

        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                                             'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                                                             'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                                             'EY2', 'F', 'G', 'HH',
                                                             'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
                                                             'M', 'N', 'NG', 'OW0', 'OW1',
                                                             'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
                                                             'UH0', 'UH1', 'UH2', 'UW',
                                                             'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}

        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        #self.cmu = cmudict.dict()
        self.homograph2features = construct_homograph_dictionary()
        self.cmudict = construct_cmu_dictionary()

    def tokenize(self, word):
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        return np.array(x)

    def predict(self, word):
        encoder = ailia.Net(weight="g2p_encoder.onnx")
        decoder = ailia.Net(weight="g2p_decoder.onnx")

        x = self.tokenize(word)

        h = encoder.run([x])[0]

        preds = []
        pred = 2    # initial symbol
        for i in range(20):
            pred = np.array([pred])
            logits, h = decoder.run([pred, h])
            pred = np.argmax(logits)
            if pred == 3: break  # 3: </s>
            preds.append(pred)

        preds = [self.idx2p.get(idx, "<unk>") for idx in preds]
        return preds

    def __call__(self, text):

        # preprocessing
        text = unicode(text)
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # tokenization
        #words = word_tokenize(text)
        #print(words)

        #print(words)
        #print(tokens)

        text2 = text
        text2 = text2.replace(".", " . ") # 句読点を単独トークンにする
        text2 = text2.replace(",", " , ")
        words = text2.split()
        #print(words2)

        #tokens = pos_tag(words)  # tuples of (word, tag)
        tokens = tag(words)

        # steps
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmudict:  # lookup CMU dict
                #if (self.cmu[word][0] != self.cmudict[word]):
                #    print(word)
                #    print(self.cmu[word][0])
                #    print(self.cmudict[word])
                #    exit()
                #pron = self.cmu[word][0] # original
                pron = self.cmudict[word] # ax impl
            else: # predict for oov
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

if __name__ == '__main__':
    texts = ["I have $250 in my pocket.", # number -> spell-out
             "popular pets, e.g. cats and dogs", # e.g. -> for example
             "I refuse to collect the refuse around here.", # homograph
             "I'm an activationist."] # newly coined word
    references = [['AY1', ' ', 'HH', 'AE1', 'V', ' ', 'T', 'UW1', ' ', 'HH', 'AH1', 'N', 'D', 'R', 'AH0', 'D', ' ', 'F', 'IH1', 'F', 'T', 'IY0', ' ', 'D', 'AA1', 'L', 'ER0', 'Z', ' ', 'IH0', 'N', ' ', 'M', 'AY1', ' ', 'P', 'AA1', 'K', 'AH0', 'T', ' ', '.'],
['P', 'AA1', 'P', 'Y', 'AH0', 'L', 'ER0', ' ', 'P', 'EH1', 'T', 'S', ' ', ',', ' ', 'F', 'AO1', 'R', ' ', 'IH0', 'G', 'Z', 'AE1', 'M', 'P', 'AH0', 'L', ' ', 'K', 'AE1', 'T', 'S', ' ', 'AH0', 'N', 'D', ' ', 'D', 'AA1', 'G', 'Z'],
['AY1', ' ', 'R', 'IH0', 'F', 'Y', 'UW1', 'Z', ' ', 'T', 'UW1', ' ', 'K', 'AH0', 'L', 'EH1', 'K', 'T', ' ', 'DH', 'AH0', ' ', 'R', 'EH1', 'F', 'Y', 'UW2', 'Z', ' ', 'ER0', 'AW1', 'N', 'D', ' ', 'HH', 'IY1', 'R', ' ', '.'],
['AY1', 'M', ' ', 'AE1', 'N', ' ', 'AE2', 'K', 'T', 'IH0', 'V', 'EY1', 'SH', 'AH0', 'N', 'IH0', 'S', 'T', ' ', '.']]
    g2p = G2p()
    for text,reference in zip(texts, references):
        out = g2p(text)
        if out != reference:
            print("Error")
            print(out)
            print(reference)
            exit()
    print("Success")
