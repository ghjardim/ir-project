import re
import nltk
import numpy as np
from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def preprocess(text=None, filename=None):
    if filename:
        f = open(filename, "r", encoding="ISO-8859-1")
        data = ""

        # Data reading and basic preprocessing
        for line in f.readlines():
            line = line.rstrip()  # Removal of trailing '\n'
            line = line.lstrip(">")  # Removal of trailing '>' quotation symbol
            data += line + " "
    else:
        data = text    

    data = data.lower()  # To lowercase

    # Converting numbers to words
    for number in re.findall(r"\d+", data):
        word = num2words(int(number))
        data = data.replace(number, word, 1)
        
    # Stopwords removal
    stop_words = stopwords.words("english")
    data_ = ""
    for word in data.split():
        if word not in stop_words:
            data_ = data_ + " " + word

    # TODO treatment of metadata separately (keep symbols)

    # Punctuation removal
    symbols = '!"#$%&()*+-./:;<=>?@[\]^_`{|}~,\n'
    for i in symbols:
        data_ = data_.replace(i, " ")

    # Stemmer
    snow_stemmer = SnowballStemmer(language="english")
    stem_words = ""
    for w in data_.split():
        x = snow_stemmer.stem(w)
        stem_words = stem_words + " " + x

    return stem_words