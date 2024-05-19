import os
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

from preprocessing import preprocess
from tfidf import Tfidf
from retrieval import compute_cosine_similarity_matrix

tfidf = Tfidf()
docs = []
docs_vect_dict = {}

pathlist = Path("data/20_newsgroups").glob("**/*")
for path in tqdm(pathlist):
    if not os.path.isdir(path):
        path_in_str = str(path)
        docs.append(preprocess(path_in_str))

docs_vect_dict = tfidf.vectorize(docs)
tfidf_matrix = tfidf.compute_tfidf_matrix()
sim_matrix = compute_cosine_similarity_matrix(tfidf_matrix)

print(sim_matrix.shape)
print(sim_matrix)
