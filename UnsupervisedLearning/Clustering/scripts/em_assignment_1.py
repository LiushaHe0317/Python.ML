import numpy
# import matplotlib.pyplot as plt
from UnsupervisedLearning.Clustering.data import DataCreator
from UnsupervisedLearning.Clustering.algorithms import ExpectationMaximize


init_means = [
    [5, 0],  # mean of cluster 1
    [1, 1],  # mean of cluster 2
    [0, 5]  # mean of cluster 3
]
init_covariances = [
    [[.5, 0.], [0, .5]],  # covariance of cluster 1
    [[.92, .38], [.38, .91]],  # covariance of cluster 2
    [[.5, 0.], [0, .5]]  # covariance of cluster 3
]
init_weights = [1 / 4., 1 / 2., 1 / 4.]  # weights of each cluster

# Generate data
data_creator = DataCreator()
numpy.random.seed(4)
data = data_creator.generate_gmm_data(init_means, init_covariances, init_weights, 100)

# plt.figure()
# d = numpy.vstack(data)
# plt.plot(d[:,0], d[:,1],'ko')
# plt.rcParams.update({'font.size':16})
# plt.tight_layout()
# plt.show()

numpy.random.seed(4)

# Initialization of parameters
chosen = numpy.random.choice(len(data), 3, replace=False)
initial_means = [data[x] for x in chosen]
initial_covs = [numpy.cov(data, rowvar=False)] * 3
initial_weights = [1/3.] * 3

# Run EM
EM = ExpectationMaximize()
results = EM.process(data, initial_means, initial_covs, initial_weights)

print(results)