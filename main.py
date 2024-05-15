import os
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

from preprocessing import preprocess
from tfidf import Tfidf

tfidf = Tfidf()
docs = []

pathlist = Path("data/20_newsgroups").glob("**/*")
for path in tqdm(pathlist):
    if not os.path.isdir(path):
        path_in_str = str(path)
        docs.append(preprocess(path_in_str))


# path_in_str = "data/20_newsgroups/rec.sport.baseball/104501"
# data =
# print(data)

pprint(tfidf.vectorize(docs))
print("\n")
print(tfidf.vocab)
print(len(tfidf.vocab))
