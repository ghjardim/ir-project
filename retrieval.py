import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity_matrix(tfidf_matrix):
    return cosine_similarity(tfidf_matrix)

def get_top_k_similar_ids(cosine_sim_matrix, id, k):
    similarities = cosine_sim_matrix[id]

    sorted_indices = np.argsort(similarities)[::-1]
    sorted_indices = sorted_indices[sorted_indices != id]

    top_k_indices = sorted_indices[:k]

    return top_k_indices.tolist()
