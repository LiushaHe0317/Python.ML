import os
from k_means import K_Means
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


kmeans = K_Means()
# set default
base_dir = r"C:\Users\heliu\Desktop\Working Directory\Clustering\dataset_01"

# normalize tf-idf word vectors
k = 3
tf_idf = kmeans.load_file(
    os.path.join(base_dir, r"people_wiki_tf_idf.npz"),
)
data = normalize(tf_idf)
heterogeneity = {}
cluster_size = {}

for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = kmeans.get_initial_centroids(data, k, seed=seed)
    centroids, cluster_assignment = kmeans.run_kmeans(data, k, initial_centroids, maxiter=400,
                                                      record_heterogeneity=None, verbose=False)

    # cluster size
    cluster_size[seed] = [data[cluster_assignment==0].shape[0],
                          data[cluster_assignment==1].shape[0],
                          data[cluster_assignment==2].shape[0]]
    print(f"seed={seed}, cluster_size={cluster_size[seed]}")

    # To save time, compute heterogeneity only once in the end
    heterogeneity[seed] = kmeans.compute_heterogeneity(data, k, centroids, cluster_assignment)
    print('seed={}, heterogeneity={}'.format(seed, heterogeneity[seed]))
