import numpy as np
import scipy as sp
import pickle
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import hnswlib

def cosine_similarity(q, A):
    print(q.shape, A.shape)
    dot_product = q.T @ A
    q_norm = np.sqrt((q ** 2).sum())
    A_norms = np.sqrt((A ** 2).sum(axis=0))
    similarities = dot_product / (q_norm * A_norms)
    return np.nan_to_num(similarities)

def cosine_similarity_normalized(q, A):
    dot_product = q.T @ A
    return dot_product

def tf_idf(matrix):
    mat = matrix.copy()
    cnt_mat = mat.count_nonzero(axis=1)
    for i in range(len(cnt_mat)):
        if cnt_mat[i] != 0:
            mat[i] *= np.log(mat.shape[1] / cnt_mat[i])
    return mat


class SearchEngine:
    def __init__(self):
        self.k = 0
        self.reduced_matrix = None
        self.svd = None
        self.p = None
        terms_by_doc = sp.sparse.load_npz("terms_by_doc.npz")
        with open("documents.json", "r") as f:
            self.documents = json.load(f)
        with open("terms.json", "r") as f:
            self.terms = json.load(f)
        with open("stop_words.txt", 'r') as sw_file:
            self.stop_words = set(sw_file.read().splitlines())
        self.stemmer = PorterStemmer()
        tfidf_transformer = TfidfTransformer()
        self.tfidf_matrix = tfidf_transformer.fit_transform(
            terms_by_doc.T).T

    def _reduce_svd_matrix(self, k):
        matrix_file = "reduced_matrix_" + str(k) + ".pkl"
        model_file = "reduced_model_" + str(k) + ".pkl"
        try:
            with open(matrix_file, 'rb') as file:
                reduced_matrix = pickle.load(file)
            with open(model_file, 'rb') as file:
                svd = pickle.load(file)
        except FileNotFoundError:
            svd = TruncatedSVD(n_components=k)
            reduced_matrix = svd.fit_transform(self.tfidf_matrix.T)
            with open(matrix_file, 'wb') as file:
                pickle.dump(reduced_matrix, file)
            with open(model_file, 'wb') as file:
                pickle.dump(svd, file)
        return reduced_matrix, svd

    def _get_ann_index(self, k):
        index_file = "ann_index_" + str(k) + ".pkl"
        try:
            with open(index_file, 'rb') as file:
                p = pickle.load(file)
        except FileNotFoundError:
            p = hnswlib.Index(space='cosine', dim=self.reduced_matrix.shape[0])
            p.init_index(max_elements=self.reduced_matrix.shape[1])
            p.set_ef(100)
            p.add_items(self.reduced_matrix.T)
            with open(index_file, 'wb') as file:
                pickle.dump(p, file)
        return p

    def search(self, search_terms: str, k: int = 0, ann: bool = False):
        query = sp.sparse.lil_matrix((len(self.terms), 1))
        words = word_tokenize(search_terms.lower())
        stemmed_words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        filtered_terms = [word for word in stemmed_words if
                          word not in self.stop_words and word.isalpha() and word in self.terms]
        if len(filtered_terms) == 0:
            print("Invalid query")
        for term in filtered_terms:
            idx = self.terms[term]
            query[idx, 0] += 1
        query *= 1 / query.sum()
        if k == 0:
            if ann:
                print("To use ANN, k has to be greater than 0. This search is performed using cosine similarity")
            fit = cosine_similarity_normalized(query, self.tfidf_matrix).toarray()[0]
            result_indices = np.argsort(-fit)
            indices = result_indices[: 20]
            accuracy = fit[indices]
        else:
            if k != self.k:
                raw_reduced_matrix, self.svd = self._reduce_svd_matrix(k)
                self.reduced_matrix = normalize(raw_reduced_matrix, norm='l2', axis=1).T
                if ann:
                    self.p = self._get_ann_index(k)
                self.k = k
            reduced_query = self.svd.transform(query.T)
            reduced_query = normalize(reduced_query, norm='l2', axis=1).T
            if ann:
                indices, distances = self.p.knn_query(reduced_query.T, k=20)
                accuracy = 1 - distances[0]
                indices = indices[0]
            else:
                fit = cosine_similarity_normalized(reduced_query, self.reduced_matrix)[0]
                result_indices = np.argsort(-fit)
                indices = result_indices[: 20]
                accuracy = fit[indices]

        results = [
            {
                "title": f"{self.documents[idx]}",
                "url": f"{self.documents[idx]}",
                "snippet": f"Match accuracy: {acc}",
            } for idx, acc in zip(indices, accuracy)]

        return results


if __name__ == '__main__':
    se = SearchEngine()
    res = se.search("computer science", k=100)
    print(res)