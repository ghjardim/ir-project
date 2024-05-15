import numpy as np
from collections import Counter


class Tfidf:

    def __init__(self):
        self.df = None
        self.vectors = None
        self.vocab = None

    def generate_vocab(self, processed_texts: list):
        df = {}

        for i in range(len(processed_texts)):
            tokens = processed_texts[i].split()
            for w in tokens:
                try:
                    df[w].add(i)
                except:
                    df[w] = {i}

        for i in df:
            df[i] = len(df[i])

        self.vocab = [x for x in df]
        self.df = df

    def vectorize(self, processed_texts: list):

        self.generate_vocab(processed_texts)
        tf_idf = {}

        for i in range(len(processed_texts)):
            tokens = processed_texts[i].split()
            counter = Counter(tokens)
            count_words = len(tokens)
            for token in np.unique(tokens):
                tf = counter[token] / count_words
                idf = np.log(len(processed_texts) / (self.df[token] + 1))
                tf_idf[i, token] = tf * idf

        self.vectors = tf_idf
        return tf_idf
