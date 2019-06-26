import math
import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv(r'data/Ads_CTR_Optimisation.csv')

# implement UCB algorithm
# global variables
N = 10000
n_dim = 10
n_select = [0] * n_dim
sum_reward = [0] * n_dim
ads_selected = []
total_reward = 0

for n in range(N):
    ad = 0
    max_upper_bound = 0
    for i in range(n_dim):
        if n_select[i] > 0:
            avg_reward = sum_reward[i] / n_select[i]
            delta_i = math.sqrt(1.5 * math.log(n + 1) / n_select[i])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i

    ads_selected.append(ad)
    n_select[ad] = n_select[ad] + 1
    reward = df.values[n, ad]
    sum_reward[ad] = sum_reward[ad] + reward
    total_reward += reward

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()