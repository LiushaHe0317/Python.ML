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

word_count = load_file(
    r"C:\Users\heliu\Desktop\ML Course\Clustering\W2_1\people_wiki_tf_idf.npz"
)

df = pd.read_csv(r"C:\Users\heliu\Desktop\ML Course\Clustering\W2_1\people_wiki.csv")
obama = df[df.name=='Barack Obama'].index.values[0]
phil = df[df.name=='Phil Schiliro'].index.values[0]

print(word_count[obama].indices[71])
print(word_count[obama].data[71])
print(word_count[phil].indices)
print(word_count[phil].data)

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

five = find_top_five(word_count[obama].indices, word_count[obama].data, word_count[phil].indices)

print(five)

count = 0
for i in range(word_count.shape[0]):
    if all(item in set(word_count[i].indices) for item in five):
        count += 1

print(count)