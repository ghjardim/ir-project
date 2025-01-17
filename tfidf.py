import numpy as np
from collections import Counter
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from ranking import compute_cosine_similarity_matrix, get_top_k_similar
from preprocessing import preprocess


class Tfidf:

    def __init__(self):
        self.df = None
        self.vectors = None
        self.vocab = None
        self.matrix = None

    def __generate_vocab(self, processed_texts: list):
        df = {}
        for i in tqdm(range(len(processed_texts)), desc="Generating vocab"):
            tokens = processed_texts[i].split()
            for w in tokens:
                try:
                    df[w].add(i)
                except:
                    df[w] = {i}
        for i in df:
            df[i] = len(df[i])
        self.vocab = [x for x in df]
        self.df = df

    def vectorize(self, processed_texts: list, id: list):
        self.__generate_vocab(processed_texts)
        self.id = id
        tf_idf = {}
        for i in tqdm(range(len(processed_texts)), desc="Vectorizing"):
            tokens = processed_texts[i].split()
            counter = Counter(tokens)
            count_words = len(tokens)
            for token in np.unique(tokens):
                tf = counter[token] / count_words
                idf = np.log((len(processed_texts) / (self.df[token])) + 1)
                tf_idf[i, token] = tf * idf
        self.vectors = tf_idf
        return tf_idf

    def get_tfidf_vector(self, doc_id):
        if self.vectors is None:
            raise ValueError("TF-IDF vectors not computed. Call vectorize() first.")
        vector = []
        for token in self.vocab:
            vector.append(self.vectors.get((doc_id, token), 0.0))
        return np.array(vector)

    def compute_tfidf_matrix(self):
        if self.vectors is None:
            raise ValueError("TF-IDF vectors not computed. Call vectorize() first.")

        num_docs = len(set(doc_id for doc_id, token in self.vectors.keys()))
        num_tokens = len(self.vocab)

        tfidf_matrix = lil_matrix((num_docs, num_tokens), dtype=np.float64)
        token_index = {token: idx for idx, token in enumerate(self.vocab)}

        for (doc_id, token), value in tqdm(
            self.vectors.items(), desc="Computing tf-idf matrix"
        ):
            tfidf_matrix[doc_id, token_index[token]] = value

        self.matrix = tfidf_matrix
        return tfidf_matrix.tocsr()

    def __query_vectorize(self, query: str):
        query_vector = []
        query_tokens = query.split()
        counter = Counter(query_tokens)
        count_words = len(query_tokens)
        for vocab_token in self.vocab:
            if vocab_token in query_tokens:
                tf = counter[vocab_token] / count_words
                # print("matrix: ", self.matrix)
                idf = np.log((self.matrix.shape[0] / (self.df[vocab_token])) + 1)
                # print(self.matrix.shape[0] / (self.df[vocab_token] + 1))
                query_vector.append(tf * idf)
            else:
                query_vector.append(0.0)
        return query_vector

    def search(
        self,
        query: str = None,
        k: int = 1,
        filename: str = None,
        is_preprocessed: bool = False,
        show_query: bool = False,
    ):
        if not is_preprocessed:
            query = preprocess(text=query, filename=filename)

        if show_query:
            print("query: ", query)

        query_vector = self.__query_vectorize(query)
        sim_mat = compute_cosine_similarity_matrix(
            docs_matrix=self.matrix, query_vector=[query_vector]
        )
        return [self.id[id] for id in get_top_k_similar(sim_mat, k)]
