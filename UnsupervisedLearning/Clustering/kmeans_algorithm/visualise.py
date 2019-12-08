import json
import os
import pandas as pd
from sklearn.preprocessing import normalize
from k_means import K_Means


kmeans = K_Means()
k = 10
base_dir = r"C:\Users\heliu\Desktop\Working Directory\Clustering\dataset_01"
df = pd.read_csv(os.path.join(base_dir, "people_wiki.csv"))
tf_idf = kmeans.load_file(
    os.path.join(base_dir, r"people_wiki_tf_idf.npz"),
)
data = normalize(tf_idf)

path_to_map = os.path.join(base_dir, r"people_wiki_map_index_to_word.json")
with open(path_to_map, 'rb') as file:
    index2words = json.load(file)

seed_list = [0, 20000, 40000, 60000, 80000, 100000, 120000]
centroids, cluster_assignment = kmeans.kmeans_multiple_runs(data, k, 400, len(seed_list), seed_list=seed_list,
                                                            verbose=True)
kmeans.visualize_document_clusters(df, data, centroids[k], cluster_assignment[k], k, index2words)