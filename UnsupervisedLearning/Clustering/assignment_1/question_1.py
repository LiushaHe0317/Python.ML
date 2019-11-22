import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def load_file(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)

doc_vectors = load_file(
    r"C:\Users\heliu\Desktop\ML Course\Clustering\W2_1\people_wiki_word_count.npz"
)

df = pd.read_csv(r"C:\Users\heliu\Desktop\ML Course\Clustering\W2_1\people_wiki.csv")
obama = df[df.name=='Barack Obama'].index.values[0]
phil = df[df.name=='Francisco Barrio'].index.values[0]

def find_top_five(idx1, data1, idx2):
    five = [0]*5
    counts = {}
    idx1_list = list(idx1)
    data1 = list(data1)
    idx2_list = list(idx2)

    for i, num in enumerate(five):
        current = num
        for j, d in enumerate(data1):
            if d > num and idx1_list[j] in set(idx2_list):
                num = d
                five[i] = idx1_list[j]
                current = num
        else:
            counts[five[i]] = current
            data1.remove(current)
            idx1_list.remove(five[i])
    return counts

five = find_top_five(doc_vectors[obama].indices, doc_vectors[obama].data, doc_vectors[phil].indices)

print(five)

count = 0
for i in range(doc_vectors.shape[0]):
    if all(item in set(doc_vectors[i].indices) for item in five):
        count += 1

print(count)