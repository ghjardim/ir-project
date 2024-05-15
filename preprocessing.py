import nltk
from nltk.corpus import stopwords
import numpy as np
from nltk.stem.snowball import SnowballStemmer

filename="./data/20_newsgroups/alt.atheism/51128"
f = open(filename, "r")
data = ""

# Data reading and basic preprocessing
for line in f.readlines():
    line = line.rstrip()    # Removal of trailing '\n'
    line = line.lstrip(">") # Removal of trailing '>' quotation symbol
    data += line + " "
data = data.lower() # To lowercase

# Stopwords removal
stop_words = stopwords.words('english')
data_ = ""
for word in data.split():
    if word not in stop_words:
        data_ = data_ + " " + word

# Stemmer
snow_stemmer = SnowballStemmer(language='english')
stem_words = ""
for w in data_.split():
    x = snow_stemmer.stem(w)
    stem_words = stem_words + " " + x

print(stem_words)

