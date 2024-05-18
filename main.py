import os
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

from preprocessing import preprocess
from tfidf import Tfidf

tfidf = Tfidf()
docs = []
docs_vect_dict = {}

pathlist = Path("data/20_newsgroups").glob("**/*")
for path in tqdm(pathlist):
    if not os.path.isdir(path):
        path_in_str = str(path)
        docs.append(preprocess(path_in_str))

docs_vect_dict = tfidf.vectorize(docs)

pprint(tfidf.get_tfidf_vector(0))
