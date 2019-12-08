import os
import sys
import numpy as np
import time
from k_means import K_Means
from sklearn.preprocessing import normalize

kmeans = K_Means()

def kmeans_multiple_runs(data, k, maxiter, num_runs, seed_list=None,
                         verbose=False):
    heterogeneity = {}

    min_heterogeneity_achieved = float('inf')
    best_seed = None
    final_centroids = None
    final_cluster_assignment = None

    for i in range(num_runs):

        # Use UTC time if no seeds are provided
        if seed_list is not None:
            seed = seed_list[i]
            np.random.seed(seed)
        else:
            seed = int(time.time())
            np.random.seed(seed)

        # Use k-means++ initialization
        initial_centroids = kmeans.get_initial_centroids(data, k, seed=seed)

        # Run k-means
        centroids, cluster_assignment = kmeans.run_kmeans(data, k, initial_centroids, maxiter,
                                                          record_heterogeneity=None, verbose=False)

        # To save time, compute heterogeneity only once in the end
        heterogeneity[seed] = kmeans.compute_heterogeneity(data, k, centroids, cluster_assignment)

        if verbose:
            print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
            sys.stdout.flush()

        # if current measurement of heterogeneity is lower than previously seen,
        # update the minimum record of heterogeneity.
        if heterogeneity[seed] < min_heterogeneity_achieved:
            min_heterogeneity_achieved = heterogeneity[seed]
            best_seed = seed
            final_centroids = centroids
            final_cluster_assignment = cluster_assignment

    return final_centroids, final_cluster_assignment

# set default
base_dir = r"C:\Users\heliu\Desktop\Working Directory\Clustering\dataset_01"
k = 3
tf_idf = kmeans.load_file(
    os.path.join(base_dir, r"people_wiki_tf_idf.npz"),
)
data = normalize(tf_idf)
seed_list = [0, 20000, 40000, 60000, 80000, 100000, 120000]

final_centroids, final_cluster_assignment = kmeans_multiple_runs(data, k, 400, len(seed_list), seed_list=seed_list,  verbose=True)

print(data[final_cluster_assignment==0].shape[0])
print(data[final_cluster_assignment==1].shape[0])
print(data[final_cluster_assignment==2].shape[0])