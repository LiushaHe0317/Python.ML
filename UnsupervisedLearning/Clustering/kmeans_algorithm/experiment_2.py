import os
import sys
import time
import matplotlib.pyplot as plt
from k_means import K_Means
from sklearn.preprocessing import normalize


kmeans = K_Means()
# set default
base_dir = r"C:\Users\heliu\Desktop\Working Directory\Clustering\dataset_01"
k = 10
tf_idf = kmeans.load_file(
    os.path.join(base_dir, r"people_wiki_tf_idf.npz"),
)
data = normalize(tf_idf)
heterogeneity_smart = {}
heterogeneity = {}

for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = kmeans.smart_initialize(data, k, seed)
    centroids, cluster_assignment = kmeans.run_kmeans(data, k, initial_centroids, maxiter=400,
                                                      record_heterogeneity=None, verbose=False)
    # To save time, compute heterogeneity only once in the end
    heterogeneity_smart[seed] = kmeans.compute_heterogeneity(data, k, centroids, cluster_assignment)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity_smart[seed]))
    sys.stdout.flush()

for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = kmeans.get_initial_centroids(data, k, seed=seed)
    centroids, cluster_assignment = kmeans.run_kmeans(data, k, initial_centroids, maxiter=400,
                                                      record_heterogeneity=None, verbose=False)

    # To save time, compute heterogeneity only once in the end
    heterogeneity[seed] = kmeans.compute_heterogeneity(data, k, centroids, cluster_assignment)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))

plt.figure(figsize=(8,5))
plt.boxplot([heterogeneity.values(), heterogeneity_smart.values()], vert=False)
plt.yticks([1, 2], ['k-means', 'k-means++'])
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.show()