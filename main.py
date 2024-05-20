import os
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

import ranking
from preprocessing import preprocess
from tfidf import Tfidf

tfidf = Tfidf()
docs = []
docs_vect_dict = {}

pathlist = Path("data/20_newsgroups").glob("**/*")
for path in tqdm(pathlist, desc="Loading docs"):
    if not os.path.isdir(path):
        path_in_str = str(path)
        docs.append(preprocess(path_in_str))

docs_vect_dict = tfidf.vectorize(docs)
tfidf_matrix = tfidf.compute_tfidf_matrix()
sim_matrix = ranking.compute_cosine_similarity_matrix(tfidf_matrix)

pprint(ranking.get_top_k_similar_ids(sim_matrix, 0, 10))
