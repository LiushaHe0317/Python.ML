import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd


def load_file(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)

tfidf = load_file(
    r"C:\Users\heliu\Desktop\ML Course\Clustering\W2_1\people_wiki_tf_idf.npz"
)

df = pd.read_csv(r"C:\Users\heliu\Desktop\ML Course\Clustering\W2_1\people_wiki.csv")

obama = tfidf[df[df.name=='Barack Obama'].index.values[0]].toarray()
biden = tfidf[df[df.name=='Joe Biden'].index.values[0]].toarray()

print(np.linalg.norm(obama-biden))
