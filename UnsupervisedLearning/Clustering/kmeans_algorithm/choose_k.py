import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from k_means import K_Means
from sklearn.preprocessing import normalize


def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, heterogeneity_values, linewidth=4)
    plt.xlabel('K')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

base_dir = r"C:\Users\heliu\Desktop\Working Directory\Clustering\dataset_01"
filename = os.path.join(base_dir, r'kmeans-arrays.npz')
kmeans = K_Means()
heterogeneity_values = []
k_list = [2, 10, 25, 50, 100]

tf_idf = kmeans.load_file(
    os.path.join(base_dir, r"people_wiki_tf_idf.npz"),
)
data = normalize(tf_idf)

if os.path.exists(filename):
    arrays = np.load(filename)
    centroids = {}
    cluster_assignment = {}
    for k in k_list:
        sys.stdout.flush()
        centroids[k] = arrays['centroids_{0:d}'.format(k)]
        cluster_assignment[k] = arrays['cluster_assignment_{0:d}'.format(k)]

        if k == 100:
            count = 0
            for j in range(100):
                if data[cluster_assignment[k]==j].shape[0] < 236:
                    count += 1
            print(count)

        score = kmeans.compute_heterogeneity(data, k, centroids[k], cluster_assignment[k])
        heterogeneity_values.append(score)

    plot_k_vs_heterogeneity(k_list, heterogeneity_values)

else:
    print('File not found. Skipping.')