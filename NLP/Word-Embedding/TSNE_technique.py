import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def wv_visualizer(we_file=None,w2i_file=None,Model=None):

    We = np.load(we_file)
    
    with open(w2i_file) as f:
        word2idx = json.load(f)

    V, D = We.shape
    idx2word = {v:k for k,v in word2idx.items()}

    model = Model()
    Z = model.fit_transform(We)

    # plot word embedding
    plt.scatter(Z[:,0],Z[:,1])
    for i in range(V):
        plt.annotate(s=idx2word[i],xy=(Z[i,0],Z[i,1]))
    plt.show()

if __name__ == '__main__':
    wv_visualizer(we_file='Data/WE.npy', w2i_file='Data/WE.json', Model=TSNE)
