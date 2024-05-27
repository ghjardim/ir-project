import os
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

import ranking
from preprocessing import preprocess
from tfidf import Tfidf

tfidf = Tfidf()
docs = []
doc_paths = []
docs_vect_dict = {}

pathlist = Path("data/20_newsgroups").glob("**/*")
for path in tqdm(pathlist, desc="Loading docs"):
    if not os.path.isdir(path):
        path_in_str = str(path)
        docs.append(preprocess(path_in_str))
        doc_paths.append(path_in_str)

docs_vect_dict = tfidf.vectorize(docs, doc_paths)
tfidf_matrix = tfidf.compute_tfidf_matrix()

print('R: ', tfidf.search("programming was the main (most time-consuming) start of the course", 4))

