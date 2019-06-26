import random
import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv(r'data/Ads_CTR_Optimisation.csv')

# implement ThomPson sampling algorithm
# global variables
N = 10000
n_dim = 10
ads_selected = []
total_reward = 0
n_reward_1 = [0] * n_dim
n_reward_0 = [0] * n_dim

for n in range(N):

    ad = 0
    max_random = 0      # maximum random draw

    for i in range(n_dim):

        random_beta = random.betavariate(n_reward_1[i] + 1, n_reward_0[i] + 1)

        if random_beta > max_random:
            max_random = random_beta
            ad = i

    # update
    ads_selected.append(ad)
    reward = df.values[n, ad]
    if reward == 1:
        n_reward_1[ad] += 1
    else:
        n_reward_0[ad] += 1
    total_reward += reward

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()