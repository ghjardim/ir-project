import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity_matrix(docs_matrix, query_vector=None):
    return cosine_similarity(query_vector, docs_matrix)

def get_top_k_similar_ids(cosine_sim_matrix, id, k):
    similarities = cosine_sim_matrix[id]
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_indices = sorted_indices[sorted_indices != id]
    top_k_indices = sorted_indices[:k]
    return top_k_indices.tolist()

def get_top_k_similar(cosine_sim_matrix, k):
    sorted_indices = np.argsort(cosine_sim_matrix[0])[::-1]
    top_k_indices = sorted_indices[:k]
    return top_k_indices.tolist()
