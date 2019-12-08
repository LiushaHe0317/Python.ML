import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import json
import os

base_dir = r"C:\Users\heliu\Desktop\ML Course\Clustering\W2_1"
path_to_map = os.path.join(base_dir, r"people_wiki_map_index_to_word.json")
path_to_tfidf = os.path.join(base_dir, r"people_wiki_tf_idf.npz")
path_to_data = os.path.join(base_dir, r"people_wiki.csv")

def load_file(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)

doc_vectors = load_file(path_to_tfidf)
df = pd.read_csv(path_to_data)
obama = df[df.name=='Barack Obama'].index.values[0]

# create a collection of random vectors
def generate_random_vectors(num_vector, dim):
    return np.random.randn(dim, num_vector)

with open(path_to_map, 'rb') as file:
    index2words = json.load(file)

# Generate 16 random vectors of dimension 547979
np.random.seed(143)
random_vectors = generate_random_vectors(num_vector=16, dim=len(index2words))

bin_bit = np.array(doc_vectors[obama].toarray()@random_vectors>=0, dtype=int)
powers_of_two = 1 << np.arange(15, -1, -1)

print("bin index = ", bin_bit@powers_of_two)
