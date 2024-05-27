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

theme_selection = None # "comp.graphics"
corpus_path = "data/20_newsgroups"
queries_path = "data/mini_newsgroups"

if theme_selection:
    corpus_path += '/'+theme_selection
    queries_path += '/'+theme_selection

pathlist = Path(corpus_path).glob("**/*")
for path in tqdm(pathlist, desc="Loading docs"):
    if not os.path.isdir(path):
        path_in_str = str(path)
        docs.append(preprocess(filename=path_in_str))
        doc_paths.append(path_in_str)

docs_vect_dict = tfidf.vectorize(docs, doc_paths)
tfidf_matrix = tfidf.compute_tfidf_matrix()

results = []

pathlist = Path(queries_path).glob("**/*")
for path in tqdm(pathlist, desc="Loading docs"):
    if not os.path.isdir(path):
        path_in_str = str(path)
        results.append(
            {
                'query': path_in_str,
                'results': tfidf.search(filename=path_in_str, k=10)
            }
        )

for res in results:
    pprint(res)
    print('----------------\n')
