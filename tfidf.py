import numpy as np
from collections import Counter


def generate_vocab(processed_texts: list):
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

    total_vocab = [x for x in df]
    return df


def tfidf(processed_texts: list):

    df = generate_vocab(processed_texts)
    tf_idf = {}

    for i in range(len(processed_texts)):
        tokens = processed_texts[i].split()
        counter = Counter(tokens)
        count_words = len(tokens)
        for token in np.unique(tokens):
            tf = counter[token] / count_words
            idf = np.log(len(processed_texts) / (df[token] + 1))
            tf_idf[i, token] = tf * idf

    return tf_idf
