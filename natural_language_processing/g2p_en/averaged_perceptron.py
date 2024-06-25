from nltk.data import load
from collections import defaultdict

class AveragedPerceptron:
    def __init__(self, weights=None):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = weights if weights else {}
        self.classes = set()

    def _softmax(self, scores):
        s = np.fromiter(scores.values(), dtype=float)
        exps = np.exp(s)
        return exps / np.sum(exps)

    def predict(self, features, return_conf=False):
        """Dot-product the features and current weights and return the best label."""
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight

        # Do a secondary alphabetic sort, for stability
        best_label = max(self.classes, key=lambda label: (scores[label], label))
        # compute the confidence
        conf = max(self._softmax(scores)) if return_conf == True else None

        return best_label, conf

model = AveragedPerceptron()

model.weights, tagdict, classes = load("averaged_perceptron_tagger.pickle")
model.classes = classes

START = ["-START-", "-START2-"]
END = ["-END-", "-END2-"]

def _get_features(i, word, context, prev, prev2):
    """Map tokens into a feature representation, implemented as a
    {hashable: int} dict. If the features change, a new model must be
    trained.
    """

    def add(name, *args):
        features[" ".join((name,) + tuple(args))] += 1

    i += len(START)
    features = defaultdict(int)
    # It's useful to have a constant feature, which acts sort of like a prior
    add("bias")
    add("i suffix", word[-3:])
    add("i pref1", word[0] if word else "")
    add("i-1 tag", prev)
    add("i-2 tag", prev2)
    add("i tag+i-2 tag", prev, prev2)
    add("i word", context[i])
    add("i-1 tag+i word", prev, context[i])
    add("i-1 word", context[i - 1])
    add("i-1 suffix", context[i - 1][-3:])
    add("i-2 word", context[i - 2])
    add("i+1 word", context[i + 1])
    add("i+1 suffix", context[i + 1][-3:])
    add("i+2 word", context[i + 2])
    return features

def normalize(word):
    """
    Normalization used in pre-processing.
    - All words are lower cased
    - Groups of digits of length 4 are represented as !YEAR;
    - Other digits are represented as !DIGITS

    :rtype: str
    """
    if "-" in word and word[0] != "-":
        return "!HYPHEN"
    if word.isdigit() and len(word) == 4:
        return "!YEAR"
    if word and word[0].isdigit():
        return "!DIGITS"
    return word.lower()

def tag(tokens, return_conf=False, use_tagdict=True):
    """
    Tag tokenized sentences.
    :params tokens: list of word
    :type tokens: list(str)
    """
    prev, prev2 = START
    output = []

    context = START + [normalize(w) for w in tokens] + END
    for i, word in enumerate(tokens):
        tag, conf = (
            (tagdict.get(word), 1.0) if use_tagdict == True else (None, None)
        )
        #print(tag, word)
        if not tag:
            features = _get_features(i, word, context, prev, prev2)
            #print(features)
            tag, conf = model.predict(features, return_conf)
        output.append((word, tag, conf) if return_conf == True else (word, tag))

        prev2 = prev
        prev = tag

    return output

#words = ["i'm", 'an', 'activationist', '.']

#output = tag(words)

#print(output)
