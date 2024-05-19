from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity_matrix(tfidf_matrix):
    return cosine_similarity(tfidf_matrix)
