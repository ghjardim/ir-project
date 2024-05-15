import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def preprocess(filename):

    f = open(filename, "r", encoding="ISO-8859-1")
    data = ""

    # Data reading and basic preprocessing
    for line in f.readlines():
        line = line.rstrip()  # Removal of trailing '\n'
        line = line.lstrip(">")  # Removal of trailing '>' quotation symbol
        data += line + " "
    data = data.lower()  # To lowercase

    # Stopwords removal
    stop_words = stopwords.words("english")
    data_ = ""
    for word in data.split():
        if word not in stop_words:
            data_ = data_ + " " + word

    # TODO treatment of metadata separately (keep symbols)

    # Punctuation removal
    symbols = '!"#$%&()*+-./:;<=>?@[\]^_`{|}~\n'
    for i in symbols:
        data_ = data_.replace(i, " ")

    # Stemmer
    snow_stemmer = SnowballStemmer(language="english")
    stem_words = ""
    for w in data_.split():
        x = snow_stemmer.stem(w)
        stem_words = stem_words + " " + x

    return stem_words
