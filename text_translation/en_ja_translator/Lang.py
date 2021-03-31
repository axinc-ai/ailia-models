
class Lang:
    """" Holds vocabulary for each language and dictionaries to convert words to and from indexes
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.word2count = {"SOS": 0, "EOS": 0}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word in self.word2count:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1

    def remove_word(self, word):
        if word != "SOS" and word != "EOS":
            self.n_words -= 1
            idx = self.word2index[word]
            last_word = self.index2word[self.n_words]

            del self.word2index[word]
            del self.word2count[word]
            del self.index2word[self.n_words]

            if last_word != word:
                self.word2index[last_word] = idx
                self.index2word[idx] = last_word
