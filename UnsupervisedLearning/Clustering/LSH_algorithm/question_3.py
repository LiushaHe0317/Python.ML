import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import json
import os
from copy import copy
from itertools import combinations

base_dir = r"C:\Users\heliu\Desktop\ML Course\Clustering\W2_1"
index2words
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
biden = df[df.name=='Joe Biden'].index.values[0]

# create a collection of random vectors
def generate_random_vectors(num_vector, dim):
    return np.random.randn(dim, num_vector)

with open(path_to_map, 'rb') as file:
    index2words = json.load(file)

# Generate 16 random vectors of dimension 547979
np.random.seed(143)
random_vectors = generate_random_vectors(num_vector=16, dim=len(index2words))

bin_bit_o = np.array(doc_vectors[obama].toarray()@random_vectors >=0, dtype=int)
bin_bit_b = np.array(doc_vectors[biden].toarray()@random_vectors >=0, dtype=int)

power_of_two = 1 << np.arange(15,-1,-1)

print('obama: ', bin_bit_o)
print('obama bin integer: ', bin_bit_o@power_of_two)
print('biden: ', bin_bit_b)
print('similarity: ', bin_bit_o@bin_bit_b.T)


def train_lsh(data, num_vector=16, seed=None):
    dim = data.shape[1]
    if seed is not None:
        np.random.seed(seed)

    random_vectors = generate_random_vectors(num_vector, dim)
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)
    table = {}

    # Partition data points into bins
    bin_index_bits = (data.dot(random_vectors) >= 0)

    # Encode bin index bits into integers
    bin_indices = bin_index_bits@powers_of_two

    # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table:
            # If no list yet exists for this bin, assign the bin an empty list.
            table[bin_index] = []
        # Fetch the list of document ids associated with the bin and add the document id to the end.
        table[bin_index].append(data_index)

    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}

    return model

model = train_lsh(doc_vectors, num_vector=16, seed=143)

table = model['table']

if 0 in table and table[0] == [39583] and 143 in table and table[143] == [19693, 28277, 29776, 30399]:
    print('Passed!')
else:
    print('Check your code.')

print (model['bin_index_bits'][35817] == model['bin_index_bits'][biden])
print (model['table'][model['bin_indices'][35817]])


def search_nearby_bins(query_bin_bits, table, search_radius=2,
                       initial_candidates=set()):
    """
    For a given query vector and trained LSH model, return all candidate neighbors for
    the query among all bins within the given search radius.
    """
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

    # Allow the user to provide an initial set of candidates.
    candidate_set = copy(initial_candidates)

    for different_bits in combinations(range(num_vector), search_radius):
        # Flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
        ## Hint: you can iterate over a tuple like a list
        alternate_bits = copy(query_bin_bits)
        for i in different_bits:
            alternate_bits[i] = ...  # YOUR CODE HERE

        # Convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)

        # Fetch the list of documents belonging to the bin indexed by the new bit vector.
        # Then add those documents to candidate_set
        # Make sure that the bin exists in the table!
        # Hint: update() method for sets lets you add an entire list to the set
        if nearby_bin in table:
            ...  # YOUR CODE HERE: Update candidate_set with the documents in this bin.

    return candidate_set

obama_bin_index = model['bin_index_bits'][35817]
candidate_set = search_nearby_bins(obama_bin_index, model['table'],
                                   search_radius=0)

if candidate_set == {35817, 21426, 53937, 39426, 50261}:
    print ('Passed test')
else:
    print ('Check your code')
print ('List of documents in the same bin as Obama: 35817, 21426, 53937, 39426, 50261')

