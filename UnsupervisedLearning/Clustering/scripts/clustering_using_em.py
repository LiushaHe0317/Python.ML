import os
import json
import numpy
import pandas as pd
from UnsupervisedLearning.Clustering.data import DataLoader
from UnsupervisedLearning.Clustering.algorithms import ExpectationMaximize
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

base_dir = r"C:\Users\heliu\Desktop\Working Directory\Clustering\wiki_dataset"
data_loader = DataLoader()

data = pd.read_csv(os.path.join(base_dir, "people_wiki.csv"))
doc_vec = data_loader.load_npz_file(os.path.join(base_dir, "4_tf_idf.npz"))
doc_vec = normalize(doc_vec)

with open(os.path.join(base_dir, "4_map_index_to_word.json"), "r") as file:
    json_data = file.read()
word2idx = json.loads(json_data)
idx2word = {v:k for k,v in word2idx.items()}

numpy.random.seed(5)
num_clusters = 25
kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=-1)
kmeans_model.fit(doc_vec)
centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_

means = [centroid for centroid in centroids]

EM = ExpectationMaximize()

num_docs = doc_vec.shape[0]
weights = []
for k in range(num_clusters):
    num_assigned = len([c for c in cluster_assignment if c == k])
    w = float(num_assigned) / num_docs
    weights.append(w)

covs = []
for k in range(num_clusters):
    member_rows = doc_vec[cluster_assignment == k]
    cov = (member_rows.multiply(member_rows) -
           2 * member_rows.dot(EM._diag(means[k]))).sum(axis=0).A1 / member_rows.shape[0] \
          + means[k] ** 2
    cov[cov < 1e-8] = 1e-8
    covs.append(cov)

out = EM.process_for_high_dimensions(doc_vec, means, covs, weights, cov_smoothing=1e-10)

num_clusters = len(means)
all_topics = set()
for c in range(num_clusters):
    print('==========================================================')
    print(f'Cluster {c}: Largest mean parameters in cluster')
    print('\n{0: <12}{1: <12}{2: <12}'.format('Word', 'Mean', 'Variance'))
    sorted_word_ids = numpy.argsort(out['means'][c])

    for i in sorted_word_ids[:5]:
        all_topics.add(idx2word[i])
        print('{0: <12}{1: <12}{2: <12}'.format(idx2word[i], out['means'][c][i], out['covs'][c][i]))

print(all_topics)

## random initialization
numpy.random.seed(5)
num_clusters = len(means)
num_docs, num_words = doc_vec.shape

random_means = []
random_covs = []
random_weights = []

for k in range(num_clusters):
    # Create a numpy array of length num_words with random normally distributed values.
    # Use the standard univariate normal distribution (mean 0, variance 1).
    mean = numpy.random.normal(0, 1, num_words)

    # Create a numpy array of length num_words with random values uniformly distributed between 1 and 5.
    cov = numpy.random.uniform(1, 5, num_words)

    # Initially give each cluster equal weight.
    weight = 1/num_clusters

    random_means.append(mean)
    random_covs.append(cov)
    random_weights.append(weight)

print(random_weights)
out_random = EM.process_for_high_dimensions(doc_vec, random_means, random_covs, random_weights, cov_smoothing=1e-10)

print('log-likelihood after k means: ', out['log-likelihood'])
print('log-likelihood after random: ', out_random['log-likelihood'])
